from typing import List


from llama_index.core.schema import (
    NodeWithScore,
    TextNode,
    MetadataMode
)




class CitationMixin():

    
    def _create_citation_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Modify retrieved nodes to be granular sources."""
        new_nodes: List[NodeWithScore] = []

        for node in nodes:
            metadata = node.metadata
            if metadata:
                text_chunk = node.get_content(metadata_mode=MetadataMode.NONE)
                if not text_chunk.startswith("Source"):
                    text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"
                else:
                    text = text_chunk

                new_node = NodeWithScore(
                    node=TextNode.parse_obj(node.node), score=node.score
                )
                new_node.node.text = text
                new_nodes.append(new_node)
            else:
                new_node = NodeWithScore(
                    node=TextNode.parse_obj(node.node), score=node.score
                )
                new_nodes.append(new_node)
        return new_nodes
