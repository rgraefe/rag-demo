import psycopg2
import psycopg2.pool
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.kvstore.postgres import PostgresKVStore


class PostgresStore():
    def __init__(self, connection_string: str, db_name:str, embedding_dim = 1024, drop_existing = False): #31.05.2024
        

        #self.connection_string = "postgresql://admin:admin@127.0.0.1:5433/vectordb"
        self.connection_string = connection_string
        #self.db_name = "vector_db"
        self.db_name = db_name
        self.embedding_dim=embedding_dim #31.05.2024


        if drop_existing:
            conn = psycopg2.connect(connection_string)
            conn.autocommit = True
            with conn.cursor() as c:
                c.execute(f"DROP DATABASE IF EXISTS {db_name}")
                c.execute(f"CREATE DATABASE {db_name}")
            
        self.url = make_url(connection_string)
        
    def set_embedding_dimension(self, embedding_dim:int):
        self.embedding_dim = embedding_dim
        
    def get_vector_store(self, table_name:str):
        return PGVectorStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=self.url.port,
            user=self.url.username,
            table_name=table_name,
            embed_dim=self.embedding_dim, #31.05.2024
        )
        
    def get_doc_store(self, table_name:str):
        return PostgresDocumentStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=self.url.port,
            user=self.url.username,
            table_name=table_name,
            namespace="document_store"
        )
        
    def get_cache_store(self, table_name:str):
        return PostgresKVStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=self.url.port,
            user=self.url.username,
            table_name=table_name
        )
        
    
