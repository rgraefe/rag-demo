from typing import Optional

import psycopg2
from psycopg2 import sql
from sqlalchemy import make_url

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.storage.kvstore.postgres import PostgresKVStore


class PostgresStore:
    def __init__(
        self,
        connection_string: str,
        db_name: str,
        embedding_dim: int = 1024,
        drop_existing: bool = False,
        tables_to_drop: Optional[list[str]] = None,
    ) -> None:
        self.connection_string = connection_string
        self.db_name = db_name
        self.embedding_dim = embedding_dim
        self.url = make_url(connection_string)

        if drop_existing:
            if not tables_to_drop:
                raise ValueError(
                    "drop_existing=True requires tables_to_drop. "
                    "This wrapper drops tables, not the whole database."
                )

            self._drop_existing_tables(tables_to_drop)

    def _drop_existing_tables(self, table_names: list[str]) -> None:
        conn = psycopg2.connect(self.connection_string)
        conn.autocommit = True

        try:
            with conn.cursor() as c:
                for table_name in table_names:
                    print(f"Dropping table if exists: {table_name}")

                    c.execute(
                        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(table_name)
                        )
                    )
        finally:
            conn.close()

    def set_embedding_dimension(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def get_vector_store(self, table_name: str) -> PGVectorStore:
        return PGVectorStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=str(self.url.port) if self.url.port else None,
            user=self.url.username,
            table_name=table_name,
            embed_dim=self.embedding_dim,
        )

    def get_doc_store(self, table_name: str) -> PostgresDocumentStore:
        return PostgresDocumentStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=str(self.url.port) if self.url.port else None,
            user=self.url.username,
            table_name=table_name,
            namespace="document_store",
        )

    def get_cache_store(self, table_name: str) -> PostgresKVStore:
        return PostgresKVStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=str(self.url.port) if self.url.port else None,
            user=self.url.username,
            table_name=table_name,
        )