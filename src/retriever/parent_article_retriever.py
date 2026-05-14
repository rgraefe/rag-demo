from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore


DEFAULT_PARENT_ID_KEY = "parent_article"
DEFAULT_NODE_LEVEL_KEY = "node_level"
DEFAULT_CHILD_LEVEL_VALUE = "paragraph"
DEFAULT_CHILD_INDEX_KEY = "article_child_index"


class ParentArticleExpansionRetriever(BaseRetriever):
    """
    Retrieves paragraph leaf nodes via vector search, then expands each hit
    to all sibling paragraphs with the same parent_article.

    This is useful when parent section/article nodes are too thin, e.g. only
    a heading line, but the paragraph children contain the useful content.
    """

    def __init__(
        self,
        leaf_retriever: BaseRetriever,
        docstore: BaseDocumentStore,
        parent_id_key: str = DEFAULT_PARENT_ID_KEY,
        node_level_key: str = DEFAULT_NODE_LEVEL_KEY,
        child_level_value: str = DEFAULT_CHILD_LEVEL_VALUE,
        child_index_key: str = DEFAULT_CHILD_INDEX_KEY,
        max_parent_articles: int = 5,
        max_siblings_per_parent: Optional[int] = None,
        return_mode: str = "merged",
    ) -> None:
        """
        Args:
            leaf_retriever:
                Your normal vector retriever over embedded paragraph nodes.

            docstore:
                The docstore containing article and paragraph nodes.

            max_parent_articles:
                Maximum number of parent articles to expand.

            max_siblings_per_parent:
                Optional limit for sibling paragraphs per parent article.

            return_mode:
                "merged" returns one merged TextNode per parent article.
                "siblings" returns the individual sibling paragraph nodes.
        """
        super().__init__()

        if return_mode not in {"merged", "siblings"}:
            raise ValueError("return_mode must be 'merged' or 'siblings'.")

        self.leaf_retriever = leaf_retriever
        self.docstore = docstore
        self.parent_id_key = parent_id_key
        self.node_level_key = node_level_key
        self.child_level_value = child_level_value
        self.child_index_key = child_index_key
        self.max_parent_articles = max_parent_articles
        self.max_siblings_per_parent = max_siblings_per_parent
        self.return_mode = return_mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 1. Normal vector retrieval over paragraph leaves
        leaf_hits = self.leaf_retriever.retrieve(query_bundle)

        if not leaf_hits:
            return []

        # 2. Group hits by parent_article
        parent_scores: Dict[str, float] = {}
        parent_hit_nodes: Dict[str, List[NodeWithScore]] = defaultdict(list)

        for hit in leaf_hits:
            parent_id = hit.node.metadata.get(self.parent_id_key)

            if not parent_id:
                continue

            score = hit.score if hit.score is not None else 0.0

            if parent_id not in parent_scores:
                parent_scores[parent_id] = score
            else:
                parent_scores[parent_id] = max(parent_scores[parent_id], score)

            parent_hit_nodes[parent_id].append(hit)

        if not parent_scores:
            return leaf_hits

        # 3. Keep strongest parent articles
        ranked_parent_ids = sorted(
            parent_scores.keys(),
            key=lambda pid: parent_scores[pid],
            reverse=True,
        )[: self.max_parent_articles]

        # 4. Fetch sibling paragraphs for each parent
        expanded_results: List[NodeWithScore] = []

        for parent_id in ranked_parent_ids:
            siblings = self._get_sibling_paragraphs(parent_id)

            if not siblings:
                # fallback: keep original vector hits
                expanded_results.extend(parent_hit_nodes[parent_id])
                continue

            siblings = self._sort_siblings(siblings)

            if self.max_siblings_per_parent is not None:
                siblings = siblings[: self.max_siblings_per_parent]

            parent_score = parent_scores[parent_id]

            if self.return_mode == "siblings":
                for sibling in siblings:
                    expanded_results.append(
                        NodeWithScore(
                            node=sibling,
                            score=parent_score,
                        )
                    )
            else:
                merged_node = self._merge_siblings(
                    parent_id=parent_id,
                    siblings=siblings,
                    hit_nodes=parent_hit_nodes[parent_id],
                )

                expanded_results.append(
                    NodeWithScore(
                        node=merged_node,
                        score=parent_score,
                    )
                )

        return expanded_results

    def _get_sibling_paragraphs(self, parent_id: str) -> List[BaseNode]:
        """
        Fetch all paragraph nodes with metadata[parent_article] == parent_id.

        This generic version scans docstore.docs. For small/medium corpora this
        is fine. For large corpora, replace this with a SQL query against your
        Postgres docstore table.
        """
        all_docs = getattr(self.docstore, "docs", None)

        if all_docs is None:
            raise RuntimeError(
                "This docstore does not expose .docs. "
                "Use the SQL-backed version below for PostgresDocumentStore."
            )

        return [
            node
            for node in all_docs.values()
            if node.metadata.get(self.parent_id_key) == parent_id
            and node.metadata.get(self.node_level_key) == self.child_level_value
        ]

    def _sort_siblings(self, siblings: Sequence[BaseNode]) -> List[BaseNode]:
        return sorted(
            siblings,
            key=lambda n: (
                n.metadata.get(self.child_index_key, 10**9),
                n.node_id,
            ),
        )

    def _merge_siblings(
        self,
        parent_id: str,
        siblings: Sequence[BaseNode],
        hit_nodes: Sequence[NodeWithScore],
    ) -> TextNode:
        """
        Create one expanded context node per parent article.
        """
        first = siblings[0]

        text_parts = []

        h1 = first.metadata.get("h1")
        h2 = first.metadata.get("h2")
        header_path = first.metadata.get("header_path")

        if header_path:
            text_parts.append(str(header_path))
            text_parts.append("")
        elif h1 or h2:
            if h1:
                text_parts.append(str(h1))
            if h2:
                text_parts.append(str(h2))
            text_parts.append("")

        for sibling in siblings:
            content = sibling.get_content().strip()
            if content:
                text_parts.append(content)

        hit_node_ids = [h.node.node_id for h in hit_nodes]

        metadata = {
            **first.metadata,
            "expanded_from_parent_article": parent_id,
            "expanded_child_count": len(siblings),
            "matched_child_node_ids": hit_node_ids,
            self.node_level_key: "expanded_article_context",
        }

        return TextNode(
            text="\n\n".join(text_parts),
            metadata=metadata,
        )