"""Parent document node parser.

Consumes output from MarkdownNodeParser and establishes parent-child
relationships for hierarchical retrieval.

Input nodes (from MarkdownNodeParser):
    node_type="section"   → article-level parent, stored in docstore only
    node_type="list_item" → leaf node, embedded, PARENT → section node

Output:
    section nodes   metadata["node_level"] = "article"
    list_item nodes metadata["node_level"] = "paragraph"
                    relationships[PARENT]  = section node id
"""

from __future__ import annotations

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

DEFAULT_PARENT_LEVEL_KEY = "node_level"
DEFAULT_PARENT_LEVEL_VAL = "article"
DEFAULT_CHILD_LEVEL_VAL  = "paragraph"
DEFAULT_PARENT_ID_KEY    = "parent_article"


class ParentDocumentNodeParser(NodeParser):
    """
    Wires parent-child relationships between section and list_item nodes
    produced by MarkdownNodeParser.

    Section nodes  → marked as "article", stored in docstore only.
    List item nodes → marked as "paragraph", embedded, PARENT → section.

    If a section has no list items it becomes both parent and leaf
    (embedded directly) so it remains searchable.

    Plugs into MyIngestionPipeline.transformations after MarkdownNodeParser.
    The pipeline's vector_node_level="paragraph" filter ensures only
    leaf nodes reach the vector store.
    """

    parent_level_key: str = Field(
        default=DEFAULT_PARENT_LEVEL_KEY,
        description="Metadata key used to mark node hierarchy level.",
    )
    parent_level_value: str = Field(
        default=DEFAULT_PARENT_LEVEL_VAL,
        description="Metadata value assigned to section-level nodes.",
    )
    child_level_value: str = Field(
        default=DEFAULT_CHILD_LEVEL_VAL,
        description="Metadata value assigned to list item leaf nodes.",
    )
    parent_id_key: str = Field(
        default=DEFAULT_PARENT_ID_KEY,
        description="Metadata key on leaf nodes storing parent section id.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "ParentDocumentNodeParser"

    @classmethod
    def from_defaults(
        cls,
        parent_level_key: str = DEFAULT_PARENT_LEVEL_KEY,
        parent_level_value: str = DEFAULT_PARENT_LEVEL_VAL,
        child_level_value: str = DEFAULT_CHILD_LEVEL_VAL,
        parent_id_key: str = DEFAULT_PARENT_ID_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "ParentDocumentNodeParser":
        return cls(
            parent_level_key=parent_level_key,
            parent_level_value=parent_level_value,
            child_level_value=child_level_value,
            parent_id_key=parent_id_key,
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
        """
        Process a sequence of nodes from MarkdownNodeParser.

        Nodes are processed in order — section nodes register themselves
        as the current parent, list_item nodes attach to the most recent
        section node with the same h2 value.
        """
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Building parent-child relationships"
        )

        # h2 → (section_node_id, has_children)
        # tracks current section and whether it has received any list items
        section_registry: dict[str, tuple[str, bool]] = {}

        result: List[BaseNode] = []

        for node in nodes_with_progress:
            node_type = node.metadata.get("node_type", "section")

            if node_type == "section":
                result.append(
                    self._process_section(node, section_registry)
                )

            elif node_type == "list_item":
                result.append(
                    self._process_list_item(node, section_registry)
                )

            else:
                # unknown type — pass through unchanged
                log.debug(
                    "Unknown node_type '%s' on node %s — passing through",
                    node_type, node.node_id,
                )
                result.append(node)

        # promote any section that received no list items to also be a leaf
        # so it remains searchable via the vector index
        result = self._promote_childless_sections(result, section_registry)

        return result

    # ── section handling ─────────────────────────────────────────────

    def _process_section(
        self,
        node: BaseNode,
        section_registry: dict[str, tuple[str, bool]],
    ) -> BaseNode:
        """
        Mark section node as article-level parent.
        Register it in section_registry keyed by h2.
        """
        h2 = node.metadata.get("h2", node.node_id)
        section_registry[h2] = (node.node_id, False)  # False = no children yet

        metadata = {
            **node.metadata,
            self.parent_level_key: self.parent_level_value,
        }
        return TextNode(
            id_=node.node_id,
            text=node.get_content(),
            metadata=metadata,
            excluded_embed_metadata_keys=list(
                set(node.excluded_embed_metadata_keys or [])
                | {self.parent_level_key}
            ),
            excluded_llm_metadata_keys=list(
                node.excluded_llm_metadata_keys or []
            ),
            relationships=dict(node.relationships or {}),
        )

    # ── list item handling ───────────────────────────────────────────

    def _process_list_item(
        self,
        node: BaseNode,
        section_registry: dict[str, tuple[str, bool]],
    ) -> BaseNode:
        """
        Mark list item as leaf node.
        Wire PARENT relationship to the matching section node.
        """
        h2 = node.metadata.get("h2", "")
        parent_id: Optional[str] = None

        if h2 in section_registry:
            parent_id, _ = section_registry[h2]
            # mark section as having at least one child
            section_registry[h2] = (parent_id, True)
        else:
            log.warning(
                "List item node %s has h2='%s' but no matching section "
                "node was found — leaf will have no parent.",
                node.node_id, h2,
            )

        metadata = {
            **node.metadata,
            self.parent_level_key: self.child_level_value,
        }
        if parent_id:
            metadata[self.parent_id_key] = parent_id

        relationships: dict = dict(node.relationships or {})
        if parent_id:
            relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=parent_id,
                metadata={self.parent_level_key: self.parent_level_value},
            )

        return TextNode(
            id_=node.node_id,
            text=node.get_content(),
            metadata=metadata,
            excluded_embed_metadata_keys=list(
                set(node.excluded_embed_metadata_keys or [])
                | {self.parent_level_key, self.parent_id_key}
            ),
            excluded_llm_metadata_keys=list(
                set(node.excluded_llm_metadata_keys or [])
                | {self.parent_level_key}
            ),
            relationships=relationships,
        )

    # ── promote childless sections ───────────────────────────────────

    def _promote_childless_sections(
        self,
        nodes: List[BaseNode],
        section_registry: dict[str, tuple[str, bool]],
    ) -> List[BaseNode]:
        """
        Any section that received no list items is promoted to also be
        a leaf — its node_level is changed to "paragraph" so the
        pipeline sends it to the vector store.

        This handles articles that have no numbered items (e.g. Article 1
        of GDPR which is just a short prose paragraph with no sub-points).
        """
        # build set of section ids that have no children
        childless_ids = {
            node_id
            for node_id, has_children in section_registry.values()
            if not has_children
        }

        if not childless_ids:
            return nodes

        result: List[BaseNode] = []
        for node in nodes:
            if (
                node.node_id in childless_ids
                and node.metadata.get(self.parent_level_key) == self.parent_level_value
            ):
                # promote: change node_level to paragraph so it gets embedded
                promoted_metadata = {
                    **node.metadata,
                    self.parent_level_key: self.child_level_value,
                }
                result.append(
                    TextNode(
                        id_=node.node_id,
                        text=node.get_content(),
                        metadata=promoted_metadata,
                        excluded_embed_metadata_keys=list(
                            set(node.excluded_embed_metadata_keys or [])
                            - {self.parent_level_key}   # allow embedding now
                        ),
                        excluded_llm_metadata_keys=list(
                            node.excluded_llm_metadata_keys or []
                        ),
                        relationships=dict(node.relationships or {}),
                    )
                )
                log.debug(
                    "Section '%s' has no list items — promoted to leaf",
                    node.metadata.get("h2", node.node_id),
                )
            else:
                result.append(node)

        return result