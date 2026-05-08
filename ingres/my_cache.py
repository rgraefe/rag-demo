from llama_index.core.ingestion import IngestionCache

from typing import List, Optional

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.utils import doc_to_json, json_to_doc
from tqdm import tqdm

class MyIngestionCache(IngestionCache):
    def put(
        self, key: str, nodes: List[BaseNode], collection: Optional[str] = None
    ) -> None:
        """Put a value into the cache."""
        collection = collection or self.collection

        val = {self.nodes_key: [doc_to_json(node) for node in tqdm(nodes, desc="Writing Cache")]}
        self.cache.put(key, val, collection=collection)

    def get(
        self, key: str, collection: Optional[str] = None
    ) -> Optional[List[BaseNode]]:
        """Get a value from the cache."""
        collection = collection or self.collection
        node_dicts = self.cache.get(key, collection=collection)

        if node_dicts is None:
            return None

        return [json_to_doc(node_dict) for node_dict in tqdm(node_dicts[self.nodes_key],desc="reading from cache")]