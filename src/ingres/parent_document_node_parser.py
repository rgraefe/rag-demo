"""Parent document node parser.

Splits article-level nodes into sub-point leaf nodes for embedding,
storing the parent relationship so the retriever can walk up to fetch
full article context for the LLM.
"""

from __future__ import annotations

import re
import hashlib
import logging
from typing import Any, Callable, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable

log = logging.getLogger(__name__)

# sub-point patterns for regulatory documents
# matches: (a), (b), (i), (ii) at the start of a line
SUBPOINT_PATTERN = re.compile(r'\n(?=\([a-z]+\)\s)')

DEFAULT_MAX_EMBED_TOKENS  = 450
DEFAULT_PARENT_LEVEL_KEY  = "node_level"
DEFAULT_PARENT_LEVEL_VAL  = "article"
DEFAULT_CHILD_LEVEL_VAL   = "paragraph"
DEFAULT_PARENT_ID_KEY     = "parent_article"


class ParentDocumentNodeParser(NodeParser):
    """
    Parent document node parser for regulatory documents.

    Takes article-level nodes and produces two types of output:
    
    1. The original article node — stored in docstore, sent to LLM.
       metadata["node_level"] = "article"
       
    2. Sub-point leaf nodes — embedded and searched.
       metadata["node_level"] = "paragraph"
       relationships[PARENT]  = article node id

    If the article fits within max_embed_tokens it is embedded as-is
    (one leaf node pointing to itself as parent). If it exceeds the
    limit it is split at sub-point boundaries.

    Plugs into MyIngestionPipeline.transformations — the pipeline's
    vector_node_level="paragraph" filter ensures only leaf nodes
    reach the vector store.
    """

    max_embed_tokens: int = Field(
        default=DEFAULT_MAX_EMBED_TOKENS,
        description="Token limit for embedding model. Nodes exceeding "
                    "this are split at sub-point boundaries.",
        gt=0,
    )
    parent_level_key: str = Field(
        default=DEFAULT_PARENT_LEVEL_KEY,
        description="Metadata key used to mark node hierarchy level.",
    )
    parent_level_value: str = Field(
        default=DEFAULT_PARENT_LEVEL_VAL,
        description="Metadata value assigned to article-level nodes.",
    )
    child_level_value: str = Field(
        default=DEFAULT_CHILD_LEVEL_VAL,
        description="Metadata value assigned to leaf/paragraph nodes.",
    )
    parent_id_key: str = Field(
        default=DEFAULT_PARENT_ID_KEY,
        description="Metadata key on leaf nodes storing parent article id.",
    )
    subpoint_pattern: str = Field(
        default=r'\n(?=\([a-z]+\)\s)',
        description="Regex pattern used to split articles into sub-points.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "ParentDocumentNodeParser"

    @classmethod
    def from_defaults(
        cls,
        max_embed_tokens: int = DEFAULT_MAX_EMBED_TOKENS,
        parent_level_key: str = DEFAULT_PARENT_LEVEL_KEY,
        parent_level_value: str = DEFAULT_PARENT_LEVEL_VAL,
        child_level_value: str = DEFAULT_CHILD_LEVEL_VAL,
        parent_id_key: str = DEFAULT_PARENT_ID_KEY,
        subpoint_pattern: str = r'\n(?=\([a-z]+\)\s)',
        include_metadata: bool = True,
        include_prev_next_rel: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "ParentDocumentNodeParser":
        return cls(
            max_embed_tokens=max_embed_tokens,
            parent_level_key=parent_level_key,
            parent_level_value=parent_level_value,
            child_level_value=child_level_value,
            parent_id_key=parent_id_key,
            subpoint_pattern=subpoint_pattern,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager or CallbackManager([]),
            id_func=id_func or default_id_func,
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
            nodes, show_progress, "Building parent-child nodes"
        )
        for node in nodes_with_progress:
            all_nodes.extend(self._process_node(node))
        return all_nodes

    # ── core logic ───────────────────────────────────────────────────

    def _process_node(self, node: BaseNode) -> List[BaseNode]:
        article_id   = node.node_id
        result: List[BaseNode] = []
        result.append(self._make_article_node(node, article_id))
        result.extend(self._make_leaf_nodes(node, article_id))
        return result

    def _make_article_node(
        self, source: BaseNode, article_id: str
    ) -> BaseNode:
        """Clone source node with article-level metadata added."""
        metadata = {
            **source.metadata,
            self.parent_level_key: self.parent_level_value,
        }
        return TextNode(
            id_=article_id,
            text=source.get_content(),
            metadata=metadata,
            # article node: exclude node_level from embedding
            excluded_embed_metadata_keys=list(
                set(source.excluded_embed_metadata_keys or [])
                | {self.parent_level_key}
            ),
            excluded_llm_metadata_keys=list(
                source.excluded_llm_metadata_keys or []
            ),
            relationships=dict(source.relationships or {}),
        )

    def _make_leaf_nodes(
        self, source: BaseNode, article_id: str
    ) -> List[BaseNode]:
        """
        Split source text into leaf nodes.

        If the text fits within max_embed_tokens: one leaf node
        pointing to the article.

        If it exceeds the limit: split at sub-point boundaries,
        one leaf per sub-point.
        """
        text = source.get_content()

        if self._token_count(text) <= self.max_embed_tokens:
            return [self._make_leaf(text, source, article_id, index=0)]

        # split at sub-point boundaries
        pattern = re.compile(self.subpoint_pattern)
        parts   = [p.strip() for p in pattern.split(text) if p.strip()]

        if len(parts) <= 1:
            # pattern didn't match anything — fall back to single leaf
            log.warning(
                "Node %s exceeds %d tokens but has no sub-point "
                "boundaries — storing as single leaf.",
                article_id, self.max_embed_tokens,
            )
            return [self._make_leaf(text, source, article_id, index=0)]

        return [
            self._make_leaf(part, source, article_id, index=i)
            for i, part in enumerate(parts)
        ]

    def _make_leaf(
        self,
        text: str,
        source: BaseNode,
        article_id: str,
        index: int,
    ) -> TextNode:
        """Build a single leaf TextNode."""
        leaf_id  = f"{article_id}_sub_{index}"
        metadata = {
            **source.metadata,
            self.parent_level_key: self.child_level_value,
            self.parent_id_key:    article_id,
        }
        return TextNode(
            id_=leaf_id,
            text=text,
            metadata=metadata,
            excluded_embed_metadata_keys=list(
                set(source.excluded_embed_metadata_keys or [])
                | {self.parent_level_key, self.parent_id_key}
            ),
            excluded_llm_metadata_keys=list(
                set(source.excluded_llm_metadata_keys or [])
                | {self.parent_level_key}
            ),
            relationships={
                NodeRelationship.PARENT: RelatedNodeInfo(
                    node_id=article_id,
                    metadata={self.parent_level_key: self.parent_level_value},
                ),
            },
        )

    # ── token counting ───────────────────────────────────────────────

    def _token_count(self, text: str) -> int:
        """
        Approximate token count.
        Uses tiktoken if available, falls back to word count × 1.3.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return int(len(text.split()) * 1.3)