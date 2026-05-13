"""Parent document node parser.

Consumes output from MyMarkdownNodeParser and establishes parent-child
relationships for hierarchical retrieval.

Input nodes from MyMarkdownNodeParser:
    node_type="section"   -> article-level parent
    node_type="list_item" -> paragraph-level leaf node

Output:
    section nodes:
        metadata["node_level"] = "article"

    list_item nodes:
        metadata["node_level"] = "paragraph"
        metadata["parent_article"] = parent section node id
        relationships[PARENT] = section node id

Notes:
    This parser returns both parent and child nodes. The later pipeline/vector
    store filtering step should decide which node levels are embedded.
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
DEFAULT_CHILD_LEVEL_VAL = "paragraph"
DEFAULT_PARENT_ID_KEY = "parent_article"


class ParentDocumentNodeParser(NodeParser):
    """
    Wires parent-child relationships between section and list_item nodes.

    Sections are registered as parent nodes.
    List items are attached to the most recent matching section using
    metadata["header_path"] as the primary key.

    If a section receives no list items, it is promoted to a leaf node by
    changing metadata["node_level"] from "article" to "paragraph", so it can
    still be embedded and retrieved.
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

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Process nodes from MyMarkdownNodeParser in order.

        section_registry maps:
            section_key -> (section_node_id, has_children)

        section_key is based on metadata["header_path"] first, then h2/h1.
        """
        nodes_with_progress = get_tqdm_iterable(
            nodes,
            show_progress,
            "Building parent-child relationships",
        )

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
                log.debug(
                    "Unknown node_type '%s' on node %s; passing through.",
                    node_type,
                    node.node_id,
                )
                result.append(node)

        return self._promote_childless_sections(result, section_registry)

    def _section_key(self, node: BaseNode) -> str:
        """
        Return stable key for parent-child matching.

        header_path is preferred because h2 alone can collide in documents
        with repeated heading names.
        """
        header_path = node.metadata.get("header_path")
        if isinstance(header_path, str) and header_path.strip():
            return header_path.strip()

        h2 = node.metadata.get("h2")
        if isinstance(h2, str) and h2.strip():
            return h2.strip()

        h1 = node.metadata.get("h1")
        if isinstance(h1, str) and h1.strip():
            return h1.strip()

        return node.node_id

    def _process_section(
        self,
        node: BaseNode,
        section_registry: dict[str, tuple[str, bool]],
    ) -> BaseNode:
        """
        Mark section node as article-level parent and register it.
        """
        section_key = self._section_key(node)
        section_registry[section_key] = (node.node_id, False)

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

    def _process_list_item(
        self,
        node: BaseNode,
        section_registry: dict[str, tuple[str, bool]],
    ) -> BaseNode:
        """
        Mark list item as leaf node and attach PARENT relationship.
        """
        section_key = self._section_key(node)
        parent_id: Optional[str] = None

        if section_key in section_registry:
            parent_id, _ = section_registry[section_key]
            section_registry[section_key] = (parent_id, True)
        else:
            log.warning(
                "List item node %s has section_key='%s' but no matching "
                "section node was found; leaf will have no parent.",
                node.node_id,
                section_key,
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
                metadata={
                    self.parent_level_key: self.parent_level_value,
                    "section_key": section_key,
                },
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

    def _promote_childless_sections(
        self,
        nodes: List[BaseNode],
        section_registry: dict[str, tuple[str, bool]],
    ) -> List[BaseNode]:
        """
        Promote sections with no list_item children to paragraph-level leaves.

        This handles short articles or sections that contain prose directly
        instead of numbered paragraphs.
        """
        childless_ids = {
            node_id
            for node_id, has_children in section_registry.values()
            if not has_children
        }

        if not childless_ids:
            return nodes

        result: List[BaseNode] = []

        for node in nodes:
            is_childless_parent = (
                node.node_id in childless_ids
                and node.metadata.get(self.parent_level_key)
                == self.parent_level_value
            )

            if not is_childless_parent:
                result.append(node)
                continue

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
                        - {self.parent_level_key}
                    ),
                    excluded_llm_metadata_keys=list(
                        node.excluded_llm_metadata_keys or []
                    ),
                    relationships=dict(node.relationships or {}),
                )
            )

            log.debug(
                "Section '%s' has no list items; promoted to leaf.",
                node.metadata.get("header_path", node.node_id),
            )

        return result