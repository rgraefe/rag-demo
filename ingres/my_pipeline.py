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
from typing import Any, List, Optional, Sequence
 
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

def run_transformations(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> List[BaseNode]:
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
        # some documents are huge and produce sub-nodes that are too many to write into the database at once
        if len(nodes) > 1000:
            batch_size = 1000
            # Calculate the number of batches
            num_batches = len(nodes) // batch_size
            remainder = len(nodes) % batch_size
            nodes_list = []
            for i in range(num_batches):
                ns = nodes[i * batch_size: (i + 1) * batch_size]
                print("Processing node-batch: {} of {}".format(i+1, num_batches))
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
                        except:
                            print("unable to write into cache")
                            pass
                        print("... end writing to cache")
                else:
                    n = transform(nodes, **kwargs)
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
                        except:
                            print("unable to write into cache")
                            pass
                        print("... end writing to cache")
                else:
                    n = transform(nodes, **kwargs)
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
                    except:
                        print("unable to write into cache")
                        pass
                    print("... end writing to cache")
            else:
                nodes = transform(nodes, **kwargs)
    return nodes


async def arun_transformations(
    nodes: List[BaseNode],
    transformations: Sequence[TransformComponent],
    in_place: bool = True,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    **kwargs: Any,
) -> List[BaseNode]:
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
            base_url=base_url,
            app_url=app_url,
            api_key=api_key,
            disable_cache=disable_cache,
        )
        
    def run(
        self,
        show_progress: bool = False,
        documents: Optional[List[Document]] = None,
        nodes: Optional[List[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        in_place: bool = True,
        store_doc_text: bool = True,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
        """
        Run a series of transformations on a set of nodes.

        If a vector store is provided, nodes with embeddings will be added to the vector store.

        If a vector store + docstore are provided, the docstore will be used to de-duplicate documents.

        Args:
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.
            documents (Optional[List[Document]], optional): Set of documents to be transformed. Defaults to None.
            nodes (Optional[List[BaseNode]], optional): Set of nodes to be transformed. Defaults to None.
            cache_collection (Optional[str], optional): Cache for transformations. Defaults to None.
            in_place (bool, optional): Whether transformations creates a new list for transformed nodes or modifies the
                array passed to `run_transformations`. Defaults to True.
            num_workers (Optional[int], optional): The number of parallel processes to use.
                If set to None, then sequential compute is used. Defaults to None.

        Returns:
            Sequence[BaseNode]: The set of transformed Nodes/Documents
        """
        input_nodes = self._prepare_inputs(documents, nodes)
        batchsize = kwargs.get("batchsize", None)
        # check if we need to dedup
        # if self.docstore is not None and self.vector_store is not None:
        #     if self.docstore_strategy in (
        #         DocstoreStrategy.UPSERTS,
        #         DocstoreStrategy.UPSERTS_AND_DELETE,
        #     ):
        #         nodes_to_run = self._handle_upserts(
        #             input_nodes, store_doc_text=store_doc_text
        #         )
        #     elif self.docstore_strategy == DocstoreStrategy.DUPLICATES_ONLY:
        #         nodes_to_run = self._handle_duplicates(
        #             input_nodes, store_doc_text=store_doc_text
        #         )
        #     else:
        #         raise ValueError(f"Invalid docstore strategy: {self.docstore_strategy}")
        # elif self.docstore is not None and self.vector_store is None:
        #     if self.docstore_strategy == DocstoreStrategy.UPSERTS:
        #         print(
        #             "Docstore strategy set to upserts, but no vector store. "
        #             "Switching to duplicates_only strategy."
        #         )
        #         self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
        #     elif self.docstore_strategy == DocstoreStrategy.UPSERTS_AND_DELETE:
        #         print(
        #             "Docstore strategy set to upserts and delete, but no vector store. "
        #             "Switching to duplicates_only strategy."
        #         )
        #         self.docstore_strategy = DocstoreStrategy.DUPLICATES_ONLY
        #     nodes_to_run = self._handle_duplicates(
        #         input_nodes, store_doc_text=store_doc_text
        #     )

        # else:
        nodes_to_run = input_nodes

        if num_workers and num_workers > 1:
            if num_workers > multiprocessing.cpu_count():
                warnings.warn(
                    "Specified num_workers exceed number of CPUs in the system. "
                    "Setting `num_workers` down to the maximum CPU count."
                )

            with multiprocessing.get_context("spawn").Pool(num_workers) as p:
                node_batches = self._node_batcher(
                    num_batches=num_workers, nodes=nodes_to_run
                )
                nodes_parallel = p.starmap(
                    run_transformations,
                    zip(
                        node_batches,
                        repeat(self.transformations),
                        repeat(in_place),
                        repeat(self.cache if not self.disable_cache else None),
                        repeat(cache_collection),
                    ),
                )
                nodes = reduce(lambda x, y: x + y, nodes_parallel, [])
            if self.vector_store is not None:
                self.vector_store.add([n for n in nodes if n.embedding is not None])
        else:
            if batchsize:
                # Calculate the number of batches
                num_batches = len(nodes_to_run) // batchsize
                remainder = len(nodes_to_run) % batchsize
                # Process batches
                for i in range(num_batches):
                    batch = nodes_to_run[i * batchsize: (i + 1) * batchsize]
                    print("Processing batch: {} of {}".format(i+1, num_batches))
                    nodes = run_transformations(
                        batch,
                        self.transformations,
                        show_progress=show_progress,
                        cache=self.cache if not self.disable_cache else None,
                        cache_collection=cache_collection,
                        in_place=in_place,
                        **kwargs,
                        )

                    if len(nodes) > 1000:
                        b_size = 1000
                        # Calculate the number of batches
                        n_batches = len(nodes) // b_size
                        r = len(nodes) % b_size
                        for i in range(n_batches):
                            ns = nodes[i * b_size: (i + 1) * b_size]
                            print("Processing node-batch for vector: {} of {}".format(i+1, n_batches))
                            if self.vector_store is not None:
                                self.vector_store.add([n for n in ns if n.embedding is not None])
                        if r > 0:
                            ns = nodes[n_batches * b_size:]
                            print("Processing node remainder vector")
                            if self.vector_store is not None:
                                self.vector_store.add([n for n in ns if n.embedding is not None])
                    else:
                        if self.vector_store is not None:
                            self.vector_store.add([n for n in nodes if n.embedding is not None])
                # Process remaining elements
                if remainder > 0:
                    batch = nodes_to_run[num_batches * batchsize:]
                    print("Processing batch remainder")
                    nodes = run_transformations(
                        batch,
                        self.transformations,
                        show_progress=show_progress,
                        cache=self.cache if not self.disable_cache else None,
                        cache_collection=cache_collection,
                        in_place=in_place,
                        **kwargs,
                        )
                                    # some documents are huge and produce sub-nodes that are too many to write into the database at once
                    if len(nodes) > 1000:
                        b_size = 1000
                        # Calculate the number of batches
                        n_batches = len(nodes) // b_size
                        remainder = len(nodes) % b_size
                        for i in range(n_batches):
                            ns = nodes[i * b_size: (i + 1) * b_size]
                            print("Processing node-batch for vector: {} of {}".format(i+1, n_batches))
                            if self.vector_store is not None:
                                self.vector_store.add([n for n in ns if n.embedding is not None])
                        if remainder > 0:
                            ns = nodes[n_batches * b_size:]
                            print("Processing node remainder vector")
                            if self.vector_store is not None:
                                self.vector_store.add([n for n in ns if n.embedding is not None])
                    else:
                        if self.vector_store is not None:
                            self.vector_store.add([n for n in nodes if n.embedding is not None])
            else:               
                nodes = run_transformations(
                    nodes_to_run,
                    self.transformations,
                    show_progress=show_progress,
                    cache=self.cache if not self.disable_cache else None,
                    cache_collection=cache_collection,
                    in_place=in_place,
                    **kwargs,
                )


                if self.vector_store is not None:
                    self.vector_store.add([n for n in nodes if n.embedding is not None])
                    

        return nodes
    
    def _handle_upserts(
        self,
        nodes: List[BaseNode],
        store_doc_text: bool = True,
    ) -> List[BaseNode]:
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
