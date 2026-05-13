from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.core.ingestion.pipeline import get_transformation_hash
 
import multiprocessing
import warnings
from functools import reduce
from itertools import repeat
from typing import Any, List, Optional
from collections.abc import Sequence
from itertools import chain
 
from llama_index.core.constants import (
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PROJECT_NAME,
)
from llama_index.core.ingestion.cache import IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import ReaderConfig
from llama_index.core.schema import (
    BaseNode,
    Document,
    TransformComponent,
)
from llama_index.core.storage.docstore import (
    BaseDocumentStore,
    SimpleDocumentStore,
)
from llama_index.core.vector_stores.types import BasePydanticVectorStore
 
from tqdm import tqdm

def is_parent_document_parser(transform: TransformComponent) -> bool:
    return (
        getattr(transform, "class_name", lambda: "")()
        == "ParentDocumentNodeParser"
    )

def run_transformations(
    nodes: Sequence[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    
    for transform in transformations:
        force_full_run = is_parent_document_parser(transform)

        if len(nodes) > 1000 and not force_full_run:
            batch_size = 1000
            num_batches = len(nodes) // batch_size
            remainder = len(nodes) % batch_size
            nodes_list = []

            for i in range(num_batches):
                ns = nodes[i * batch_size: (i + 1) * batch_size]
                print("Processing node-batch: {} of {}".format(i + 1, num_batches))

                if cache is not None:
                    hash = get_transformation_hash(ns, transform)
                    cached_nodes = cache.get(hash, collection=cache_collection)

                    if cached_nodes is not None:
                        n = cached_nodes
                    else:
                        n = transform(ns, **kwargs)
                        print("writing to cache ...")
                        try:
                            cache.put(hash, n, collection=cache_collection)
                        except Exception:
                            print("unable to write into cache")
                        print("... end writing to cache")
                else:
                    n = transform(ns, **kwargs)

                nodes_list.extend(n)

            if remainder > 0:
                ns = nodes[num_batches * batch_size:]
                print("Processing node remainder")

                if cache is not None:
                    hash = get_transformation_hash(ns, transform)
                    cached_nodes = cache.get(hash, collection=cache_collection)

                    if cached_nodes is not None:
                        n = cached_nodes
                    else:
                        n = transform(ns, **kwargs)
                        print("writing to cache ...")
                        try:
                            cache.put(hash, n, collection=cache_collection)
                        except Exception:
                            print("unable to write into cache")
                        print("... end writing to cache")
                else:
                    n = transform(ns, **kwargs)

                nodes_list.extend(n)

            nodes = nodes_list

        else:
            if cache is not None:
                hash = get_transformation_hash(nodes, transform)
                cached_nodes = cache.get(hash, collection=cache_collection)

                if cached_nodes is not None:
                    nodes = cached_nodes
                else:
                    nodes = transform(nodes, **kwargs)
                    print("writing to cache ...")
                    try:
                        cache.put(hash, nodes, collection=cache_collection)
                    except Exception:
                        print("unable to write into cache")
                    print("... end writing to cache")
            else:
                nodes = transform(nodes, **kwargs)
    return nodes


async def arun_transformations(
    nodes: Sequence[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """Run a series of transformations on a set of nodes.

    Args:
        nodes: The nodes to transform.
        transformations: The transformations to apply to the nodes.

    Returns:
        The transformed nodes.
    """
    if not in_place:
        nodes = list(nodes)

    for transform in transformations:
        if cache is not None:
            hash = get_transformation_hash(nodes, transform)

            cached_nodes = cache.get(hash, collection=cache_collection)
            if cached_nodes is not None:
                nodes = cached_nodes
            else:
                nodes = await transform.acall(nodes, **kwargs)
                cache.put(hash, nodes, collection=cache_collection)
        else:
            nodes = await transform.acall(nodes, **kwargs)

    return nodes


class MyIngestionPipeline(IngestionPipeline):
    def __init__(
        self,
        name: str = DEFAULT_PIPELINE_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        transformations: Optional[List[TransformComponent]] = None,
        readers: Optional[List[ReaderConfig]] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        cache: Optional[IngestionCache] = None,
        docstore: Optional[BaseDocumentStore] = None,
        docstore_strategy: DocstoreStrategy = DocstoreStrategy.UPSERTS,
        base_url: Optional[str] = None,
        app_url: Optional[str] = None,
        api_key: Optional[str] = None,
        disable_cache: bool = False,
    ) -> None:

        super().__init__(
            name=name,
            project_name=project_name,
            transformations=transformations,
            readers=readers,
            documents=documents,
            vector_store=vector_store,
            cache=cache or IngestionCache(),
            docstore=docstore,
            docstore_strategy=docstore_strategy,
            disable_cache=disable_cache,
        )

    def run(
        self,
        show_progress: bool = False,
        documents: Optional[Sequence[Document]] = None,
        nodes: Optional[Sequence[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        store_doc_text: bool = True,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        """
        Run ingestion in two phases:

        1. Structural phase:
        - MyMarkdownNodeParser
        - ParentDocumentNodeParser
        Writes ALL structural nodes to docstore:
        - article parent nodes
        - paragraph leaf nodes

        2. Embedding phase:
        - ParagraphOnlyFilter
        - embed_model
        Writes only embedded leaf nodes to vector store.
        """

        input_nodes = list(self._prepare_inputs(documents, nodes))

        if not input_nodes:
            return []

        # ------------------------------------------------------------
        # 1. Split transformations into:
        #    structural_transformations = up to and including ParentDocumentNodeParser
        #    embedding_transformations = everything after it
        # ------------------------------------------------------------

        parent_parser_index: Optional[int] = None

        for i, transform in enumerate(self.transformations):
            if is_parent_document_parser(transform):
                parent_parser_index = i
                break

        if parent_parser_index is None:
            # Fallback: no ParentDocumentNodeParser found.
            # Behave like a normal ingestion pipeline, but still use your
            # batching/vector-store logic.
            result_nodes = list(
                run_transformations(
                    input_nodes,
                    self.transformations,
                    show_progress=show_progress,
                    cache=self.cache if not self.disable_cache else None,
                    cache_collection=cache_collection,
                    in_place=in_place,
                    **kwargs,
                )
            )

            if self.vector_store is not None:
                embedded_nodes = [n for n in result_nodes if n.embedding is not None]
                if embedded_nodes:
                    self.vector_store.add(embedded_nodes)

            if self.docstore is not None:
                self.docstore.set_document_hashes(
                    {n.id_: n.hash for n in result_nodes}
                )
                self.docstore.add_documents(
                    result_nodes,
                    store_text=store_doc_text,
                )

            return result_nodes

        structural_transformations = self.transformations[: parent_parser_index + 1]
        embedding_transformations = self.transformations[parent_parser_index + 1 :]

        # ------------------------------------------------------------
        # 2. Deduplicate source documents before running transformations
        # ------------------------------------------------------------

        effective_strategy = self.docstore_strategy

        if (
            self.docstore is not None
            and self.vector_store is None
            and self.docstore_strategy
            in (DocstoreStrategy.UPSERTS, DocstoreStrategy.UPSERTS_AND_DELETE)
        ):
            warnings.warn(
                f"docstore_strategy='{self.docstore_strategy.value}' requires "
                "a vector store to apply upsert/delete semantics; falling back "
                "to 'duplicates_only' for this run.",
                UserWarning,
                stacklevel=2,
            )
            effective_strategy = DocstoreStrategy.DUPLICATES_ONLY

        if self.docstore is not None and self.vector_store is not None:
            if effective_strategy in (
                DocstoreStrategy.UPSERTS,
                DocstoreStrategy.UPSERTS_AND_DELETE,
            ):
                nodes_to_run = self._handle_upserts(input_nodes)
            elif effective_strategy == DocstoreStrategy.DUPLICATES_ONLY:
                nodes_to_run = self._handle_duplicates(input_nodes)
            else:
                raise ValueError(f"Invalid docstore strategy: {effective_strategy}")

        elif self.docstore is not None and self.vector_store is None:
            nodes_to_run = self._handle_duplicates(input_nodes)

        else:
            nodes_to_run = input_nodes

        if not nodes_to_run:
            return []

        # ------------------------------------------------------------
        # 3. Run structural transformations
        #    Important: ParentDocumentNodeParser must see the full ordered stream.
        # ------------------------------------------------------------

        structural_nodes = list(
            run_transformations(
                nodes_to_run,
                structural_transformations,
                show_progress=show_progress,
                cache=self.cache if not self.disable_cache else None,
                cache_collection=cache_collection,
                in_place=in_place,
                **kwargs,
            )
        )

        # ------------------------------------------------------------
        # 4. Write ALL structural nodes to docstore
        #    This includes:
        #    - article parent nodes
        #    - paragraph leaf nodes
        # ------------------------------------------------------------

        if self.docstore is not None:
            self.docstore.set_document_hashes(
                {n.id_: n.hash for n in structural_nodes}
            )
            self.docstore.add_documents(
                structural_nodes,
                store_text=store_doc_text,
            )

        # ------------------------------------------------------------
        # 5. Run embedding transformations
        #    Usually:
        #    - ParagraphOnlyFilter()
        #    - embed_model
        # ------------------------------------------------------------

        if embedding_transformations:
            embedded_nodes = list(
                run_transformations(
                    structural_nodes,
                    embedding_transformations,
                    show_progress=show_progress,
                    cache=self.cache if not self.disable_cache else None,
                    cache_collection=cache_collection,
                    in_place=in_place,
                    **kwargs,
                )
            )
        else:
            embedded_nodes = structural_nodes

        # ------------------------------------------------------------
        # 6. Write only embedded nodes to vector store
        # ------------------------------------------------------------

        if self.vector_store is not None:
            nodes_with_embeddings = [
                n for n in embedded_nodes
                if n.embedding is not None
            ]

            if nodes_with_embeddings:
                b_size = 1000

                for start in range(0, len(nodes_with_embeddings), b_size):
                    batch = nodes_with_embeddings[start : start + b_size]

                    print(
                        f"Processing node-batch for vector: "
                        f"{start // b_size + 1} of "
                        f"{(len(nodes_with_embeddings) + b_size - 1) // b_size}"
                    )

                    self.vector_store.add(batch)

        return embedded_nodes
        
    def _handle_upserts(
        self,
        nodes: Sequence[BaseNode],
        store_doc_text: bool = True,
    ) -> Sequence[BaseNode]:
        """Handle docstore upserts by checking hashes and ids."""
        assert self.docstore is not None
        print("getting existing doc ids")
        existing_doc_ids_before = set(self.docstore.get_all_document_hashes().values())
        doc_ids_from_nodes = set()
        deduped_nodes_to_run = {}
        for node in tqdm(nodes,desc="Handling upserts"):
            ref_doc_id = node.ref_doc_id if node.ref_doc_id else node.id_
            doc_ids_from_nodes.add(ref_doc_id)
            existing_hash = self.docstore.get_document_hash(ref_doc_id)
            if not existing_hash:
                # document doesn't exist, so add it
                self.docstore.set_document_hash(ref_doc_id, node.hash)
                deduped_nodes_to_run[ref_doc_id] = node
            elif existing_hash and existing_hash != node.hash:
                self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)

                if self.vector_store is not None:
                    self.vector_store.delete(ref_doc_id)

                self.docstore.set_document_hash(ref_doc_id, node.hash)

                deduped_nodes_to_run[ref_doc_id] = node
            else:
                continue  # document exists and is unchanged, so skip it

        if self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
            # Identify missing docs and delete them from docstore and vector store
            doc_ids_to_delete = existing_doc_ids_before - doc_ids_from_nodes
            for ref_doc_id in doc_ids_to_delete:
                self.docstore.delete_document(ref_doc_id)

                if self.vector_store is not None:
                    self.vector_store.delete(ref_doc_id)

        nodes_to_run = list(deduped_nodes_to_run.values())
        print("Writing hashes to docstore ...")
        self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)
        print("... finished writing to docstore")

        return nodes_to_run
