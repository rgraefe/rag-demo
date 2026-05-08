from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

from llama_index.core import Settings
import asyncio
import logging
import sys
import argparse
sys.path.append('../')
from ingres import ReaderFactory, MyIngestionPipeline, MyIngestionCache, MySemanticNodeParser
from models import Modeltypes, ModelFactory
from database import PostgresStore
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionCache,
)
from tqdm import tqdm
import json
from preprocessing.preprocessing import TextCleaner1st

# logging.config.fileConfig('/home/rgraefe/git/rag_chat/log.conf')
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#Linux
# arg_dict = {"rootdir":"/data/LLM", "treeroots": ["*_index.md", "[A-Za-z]*.mmd"],
#             "exclude":["*.pu","*.puml","*Template.docx","_cache/*","_db_backup/*","_sums/*","__db_backup/*","ras-tables.xlsx"], 
#             "sum_path":"/data/LLM/_sums", "cached_documents": True}

#Windows
arg_dict = {"rootdir":"C:\\Users\\rgraefe\\OneDrive - Intel Corporation\\Documents\\LLM", "treeroots": ["*_index.md", "[A-Za-z]*.mmd"],
            "exclude":["*.pu","*.puml","*Template.docx","_cache/*","_db_backup/*","_sums/*","__db_backup/*","ras-tables.xlsx"], 
            "sum_path":"C:\\Users\\rgraefe\\OneDrive - Intel Corporation\\Documents\\LLM\\_sums", "cached_documents": False}

args = argparse.Namespace(**arg_dict)

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

def find_non_serializable_elements(obj, path=""):
    non_serializable_elements = []
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_path = f"{path}/{k}"
            if not is_json_serializable(v):
                non_serializable_elements.append(full_path)
            non_serializable_elements.extend(find_non_serializable_elements(v, full_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            full_path = f"{path}[{i}]"
            if not is_json_serializable(item):
                non_serializable_elements.append(full_path)
            non_serializable_elements.extend(find_non_serializable_elements(item, full_path))
    else:
        if not is_json_serializable(obj):
            non_serializable_elements.append(path)
    
    return non_serializable_elements

def read_documents(args):
    root_dir = args.rootdir
    treeroot = args.treeroots
    exclude = args.exclude
    cached_documents = args.cached_documents
    if len(treeroot) > 0:
        reader = ReaderFactory(input_dir=root_dir, recursive=True, exclude=exclude, tree_root_glob=treeroot, 
                                cached_documents=cached_documents)
    else:
        reader = ReaderFactory(input_dir=root_dir, recursive=True, exclude=exclude,cached_documents=cached_documents)
    return reader.load_data()

sum_path = args.sum_path
cached_documents = args.cached_documents

chatllm = ModelFactory.getLlmModel(modeltype=Modeltypes.AZURE)
sumllm = ModelFactory.getLlmModel(modeltype=Modeltypes.OLLAMA)
#sumllm = chatllm
embed_model = ModelFactory.getEmbedModel(modeltype=Modeltypes.NOMIC)

Settings.embed_model = embed_model
Settings.llm = sumllm
sum_llm =sumllm
#docs = read_documents(args=args)

storage = PostgresStore("postgresql://admin:admin@10.62.197.158:5433/vectordb", "vector_db")


docstore = storage.get_doc_store("d_nodes_index")
deduped_nodes_to_run = []

# for doc in tqdm(docs, "storing docs to database"):
#     if docstore.document_exists(doc.id_):
#       continue  # document exists and is unchanged, so skip it
#     # if doc.id_ != "9f558e7f-6277-4823-9d3f-f6a9b9946b33":
#     #     continue
#     docstore.add_documents([doc,])
docs = list(docstore.docs.values())





# window_node_parser = SentenceWindowNodeParser.from_defaults(
#     # how many sentences on either side to capture
#     window_size=3,
#     # the metadata key that holds the window of surrounding sentences
#     window_metadata_key="window",
#     # the metadata key that holds the original sentence
#     original_text_metadata_key="original_sentence",
#     include_prev_next_rel=True,
#     include_metadata=True
# )

#sentence_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=100, include_metadata=True, include_prev_next_rel=True)
semantic_parser = MySemanticNodeParser.from_defaults(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model,exclude_metadata_tags=["table_df"])

storage.set_embedding_dimension(embedding_dim=768)

indexed_pipeline = MyIngestionPipeline(
    transformations=[TextCleaner1st(),
                     #semantic_parser,
                     embed_model,
        
    ],
    docstore=storage.get_doc_store("d_nodes_index_clean"),
    vector_store=storage.get_vector_store("v_nodes_index_clean"),
    cache=MyIngestionCache(
        cache=storage.get_cache_store("c_nodes_index_clean"),
        collection="w_nodes_index_cache",
    ),
    docstore_strategy=DocstoreStrategy.UPSERTS,
)

# sentence_pipeline = MyIngestionPipeline(
#     transformations=[TextCleaner1st(),
#                      sentence_node_parser,
#                      embed_model,
        
#     ],
#     docstore=storage.get_doc_store("d_sentence_split_clean"),
#     vector_store=storage.get_vector_store("v_sentence_split_clean"),
#     cache=IngestionCache(
#         cache=storage.get_cache_store("c_sentence_split_clean"),
#         collection="w_postgres_cache",
#     ),
#     docstore_strategy=DocstoreStrategy.UPSERTS,
# )

# read the elements in batches
batch_size = 32
# Calculate the number of batches
num_batches = len(docs) // batch_size
remainder = len(docs) % batch_size

# async def process_batches():
#     # Process batches
#     for i in range(num_batches):
#         batch = docs[i * batch_size: (i + 1) * batch_size]
#         print("Processing batch: {} of {}".format(i+1, num_batches))
#         window_nodes = await window_pipeline.arun(documents=batch,show_progress=True)
#         sentence_nodes = await sentence_pipeline.arun(documents=batch,show_progress=True)

#     # Process remaining elements
#     if remainder > 0:
#         remaining_batch = docs[num_batches * batch_size:]
#         print("Processing batch: {} of {}".format(i+1, num_batches))
#         window_nodes = window_pipeline.arun(documents=batch,show_progress=True)
#         sentence_nodes = sentence_pipeline.arun(documents=batch,show_progress=True)

# asyncio.run(process_batches())


indexed_nodes = indexed_pipeline.run(documents=docs,show_progress=True, batchsize=batch_size)
#sentence_nodes = sentence_pipeline.run(documents=docs,show_progress=True, batchsize=batch_size)

