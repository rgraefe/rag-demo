from .citation_mixin import CitationMixin
from llama_index.core.response_synthesizers import Accumulate
from typing import Any, Optional
import llama_index.core.instrumentation as instrument
from typing import Any, List, Optional, Sequence
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
)
from llama_index.core.schema import (
    NodeWithScore,
    QueryType,
)

QueryTextType = QueryType
dispatcher = instrument.get_dispatcher(__name__)

class AccumulateCite(Accumulate, CitationMixin):
        
    @dispatcher.span
    def synthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        
        cite_nodes = self._create_citation_nodes(nodes)
        return super().synthesize(
            query=query,
            nodes=cite_nodes,
            additional_source_nodes=additional_source_nodes,
            response_kwargs=response_kwargs
        )

    @dispatcher.span
    async def asynthesize(
        self,
        query: QueryTextType,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
        **response_kwargs: Any,
    ) -> RESPONSE_TYPE:
        
        cite_nodes = self._create_citation_nodes(nodes)
        return super().asynthesize(
            query=query,
            nodes=cite_nodes,
            additional_source_nodes=additional_source_nodes,
            response_kwargs=response_kwargs
        )