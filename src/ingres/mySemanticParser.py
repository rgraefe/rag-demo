from llama_index.core.node_parser import SemanticSplitterNodeParser
from typing import Callable, List, Optional, Sequence
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.node_parser.text.utils import split_by_sentence_tokenizer
from llama_index.core.schema import BaseNode, Document, TextNode, IndexNode
from llama_index.core import Settings
from src.utils import document_to_node
from llama_index.core.bridge.pydantic import Field

DEFAULT_OG_TEXT_METADATA_KEY = "original_text"

class MySemanticNodeParser(SemanticSplitterNodeParser):
    
    exclude_metadata_tags: List[str] = Field(
        default=[],
        description="Metadata tags to exclude from splitting.",
    )
    
    original_text_metadata_key: str = Field(
        default=DEFAULT_OG_TEXT_METADATA_KEY,
        description="Key for original text metadata.",
    )
    
    @classmethod
    def from_defaults(
        cls,
        embed_model: Optional[BaseEmbedding] = None,
        breakpoint_percentile_threshold: Optional[int] = 95,
        buffer_size: Optional[int] = 1,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
        exclude_metadata_tags: Optional[List[str]] = None
    ) -> "SemanticSplitterNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        sentence_splitter = sentence_splitter or split_by_sentence_tokenizer()
        if embed_model is None:
            if Settings.embed_model:
                embed_model = Settings.embed_model
                
            else:
                try:
                    from llama_index.embeddings.openai import (
                        OpenAIEmbedding,
                    )  # pants: no-infer-dep

                    embed_model = embed_model or OpenAIEmbedding()
                except ImportError:
                    raise ImportError(
                        "`llama-index-embeddings-openai` package not found, "
                        "please run `pip install llama-index-embeddings-openai`"
                    )

        id_func = id_func or default_id_func

        breakpoint_percentile_threshold = breakpoint_percentile_threshold or 95
        buffer_size = buffer_size or 1
        exclude_metadata_tags = exclude_metadata_tags or []

        return cls(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size,
            sentence_splitter=sentence_splitter,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
            exclude_metadata_tags=exclude_metadata_tags
        )
        


    def build_semantic_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Build window nodes from documents."""
        all_nodes: List[BaseNode] = []
        for doc in documents:
            # prevent splitting for some nodes e.g. tables
            if isinstance(doc, IndexNode):
                all_nodes.append(doc)
                continue
            if self.exclude_metadata_tags:
                for tag in self.exclude_metadata_tags:
                    metadata = doc.metadata
                    if tag in metadata.keys():
                        if isinstance(doc, Document):
                            doc = document_to_node(doc)
                        all_nodes.append(doc)
                        continue
            text = doc.text
            text_splits = self.sentence_splitter(text)

            sentences = self._build_sentence_groups(text_splits)

            combined_sentence_embeddings = self.embed_model.get_text_embedding_batch(
                [s["combined_sentence"] for s in sentences],
                show_progress=show_progress,
            )

            for i, embedding in enumerate(combined_sentence_embeddings):
                sentences[i]["combined_sentence_embedding"] = embedding

            distances = self._calculate_distances_between_sentence_groups(sentences)

            chunks = self._build_node_chunks(sentences, distances)

            nodes = build_nodes_from_splits(
                chunks,
                doc,
                id_func=self.id_func,
            )

            all_nodes.extend(nodes)

        return all_nodes