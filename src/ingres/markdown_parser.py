"""
markdown_node_parser.py
=======================
Splits a markdown document into nodes by:
  - H1 / H2 headings  (## Article 17)
  - Extra heading rules (e.g. bare "Article 17" lines in PDFs)
  - Top-level numbered list items (1. 2. 3.)

Each node carries:
  metadata["h1"]          - current H1 heading text
  metadata["h2"]          - current H2 heading text (if any)
  metadata["header_path"] - "/h1/h2" style path
  metadata["node_type"]   - "section" | "list_item"
  metadata["list_index"]  - int, only for list_item nodes
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable

from src.ingres.heading_rules import HeadingRule

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

@dataclass
class _ParserState:
    """Mutable state carried through the line loop."""
    h1:           str = ""
    h2:           str = ""
    buffer:       list[str] = field(default_factory=list)
    node_type:    str = "section"      # "section" | "list_item"
    list_index:   int = 0
    in_code_block: bool = False

    def flush(self) -> Optional[str]:
        """Return buffered text and clear buffer. None if buffer is empty."""
        text = "\n".join(self.buffer).strip()
        self.buffer = []
        return text if text else None

    @property
    def header_path(self) -> str:
        parts = [p for p in [self.h1, self.h2] if p]
        return "/" + "/".join(parts) + "/" if parts else "/"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class MyMarkdownNodeParser(NodeParser):
    """
    Splits a markdown document into nodes.

    Splitting triggers:
      1. H1 heading  (# ...)           → new node, resets h1 + h2
      2. H2 heading  (## ...)          → new node, resets h2
      3. Extra heading rules            → treated as H1 or H2
      4. Top-level numbered list item   → new node per item

    Everything else (body text, sub-points (a)(b), tables) stays
    in the buffer and becomes part of the current node.
    """

    heading_rules: List[Any] = Field(
        default_factory=list,
        description="Extra HeadingRule objects for lines not marked with #.",
    )

    @classmethod
    def from_defaults(
        cls,
        heading_rules: Optional[List[HeadingRule]] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "MyMarkdownNodeParser":
        return cls(
            heading_rules=heading_rules or [],
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager or CallbackManager([]),
        )

    # ── NodeParser interface ─────────────────────────────────────────

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing markdown nodes"
        )
        for node in nodes_with_progress:
            all_nodes.extend(self.get_nodes_from_node(node))
        return all_nodes

    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Split a single node's text into section/list_item nodes."""
        text  = node.get_content(metadata_mode=MetadataMode.NONE)
        lines = text.splitlines()
        state = _ParserState()
        result: List[TextNode] = []

        for line in lines:

            # ── code block tracking (never split inside a code block) ──
            if line.lstrip().startswith("```"):
                state.in_code_block = not state.in_code_block
                state.buffer.append(line)
                continue

            if state.in_code_block:
                state.buffer.append(line)
                continue

            # ── H1 heading ───────────────────────────────────────────
            h1_match = re.match(r"^#\s+(.+)", line)
            if h1_match:
                self._flush_node(state, node, result)
                state.h1         = h1_match.group(1).strip()
                state.h2         = ""
                state.node_type  = "section"
                state.buffer     = [line]
                continue

            # ── H2 heading ───────────────────────────────────────────
            h2_match = re.match(r"^##\s+(.+)", line)
            if h2_match:
                self._flush_node(state, node, result)
                state.h2        = h2_match.group(1).strip()
                state.node_type = "section"
                state.buffer    = [line]
                continue

            # ── extra heading rules ──────────────────────────────────
            rule_match = self._match_heading_rule(line)
            if rule_match is not None:
                rule_level, heading_text = rule_match
                self._flush_node(state, node, result)
                if rule_level == 1:
                    state.h1 = heading_text
                    state.h2 = ""
                else:
                    state.h2 = heading_text
                state.node_type = "section"
                state.buffer    = [line]
                continue

            # ── top-level numbered list item ─────────────────────────
            list_match = re.match(r"^(\d+)\.\s+(.+)", line)
            if list_match:
                self._flush_node(state, node, result)
                state.list_index = int(list_match.group(1))
                state.node_type  = "list_item"
                state.buffer     = [line]
                continue

            # ── everything else → append to current buffer ───────────
            state.buffer.append(line)

        # flush final buffer
        self._flush_node(state, node, result)
        return result

    # ── helpers ──────────────────────────────────────────────────────

    def _match_heading_rule(
        self, line: str
    ) -> Optional[tuple[int, str]]:
        """Check line against all heading rules. Returns (level, text) or None."""
        if line.startswith("#"):
            return None
        for rule in self.heading_rules:
            m = rule.pattern.match(line)
            if m:
                text = rule.pattern.sub("", line).strip() if rule.strip_match else line.strip()
                return rule.level, text
        return None

    def _flush_node(
        self,
        state: _ParserState,
        source: BaseNode,
        result: List[TextNode],
    ) -> None:
        """Flush buffer to a TextNode and append to result."""
        text = state.flush()
        if not text:
            return

        metadata: dict = {
            **source.metadata,
            "header_path": state.header_path,
            "node_type":   state.node_type,
        }
        if state.h1:
            metadata["h1"] = state.h1
        if state.h2:
            metadata["h2"] = state.h2
        if state.node_type == "list_item":
            metadata["list_index"] = state.list_index

        nodes = build_nodes_from_splits(
            [text], source, id_func=self.id_func
        )
        if nodes:
            nodes[0].metadata = metadata
            result.append(nodes[0])