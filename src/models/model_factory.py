from dotenv import load_dotenv, find_dotenv
import logging
import os
from enum import Enum

from llama_index.core.llms.llm import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

log = logging.getLogger(__name__)


class Modeltypes(Enum):
    OPENAI = 1
    OLLAMA = 2
    NOMIC = 3
    SNOWFLAKE = 4
    MXBAI = 5


class ModelFactory:
    """Factory for LLM and embedding models."""

    @staticmethod
    def getLlmModel(modeltype: Modeltypes) -> LLM:
        if modeltype == Modeltypes.OPENAI:
            return ModelFactory._getOpenAILlmModel()

        if modeltype == Modeltypes.OLLAMA:
            return ModelFactory._getOllamaLlmModel()

        raise ValueError(f"Unsupported LLM model type: {modeltype}")

    @staticmethod
    def getEmbedModel(modeltype: Modeltypes) -> BaseEmbedding:
        if modeltype == Modeltypes.OPENAI:
            return ModelFactory._getOpenAIEmbedModel()

        if modeltype == Modeltypes.NOMIC:
            return ModelFactory._getNomicEmbedModel()

        if modeltype == Modeltypes.MXBAI:
            return ModelFactory._getMxbaiEmbedModel()

        if modeltype == Modeltypes.SNOWFLAKE:
            return ModelFactory._getSnowflakeEmbedModel()

        raise ValueError(f"Unsupported embedding model type: {modeltype}")

    @staticmethod
    def _getOpenAILlmModel() -> LLM:
        """Return OpenAI LLM.

        Expects OPENAI_API_KEY to be provided through a .env file.
        """
        load_dotenv(find_dotenv())

        return OpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )

    @staticmethod
    def _getOllamaLlmModel() -> LLM:
        """Return Ollama LLM."""
        return Ollama(
            model="llama3.2",
            base_url="http://localhost:11434",
            request_timeout=60.0,
        )

    @staticmethod
    def _getOpenAIEmbedModel() -> BaseEmbedding:
        """Return OpenAI embedding model.

        Expects OPENAI_API_KEY to be provided through a .env file.
        """
        load_dotenv(find_dotenv())

        return OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"],
        )

    @staticmethod
    def _getNomicEmbedModel() -> BaseEmbedding:
        """Return Ollama embedding for nomic-embed-text."""
        return OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )

    @staticmethod
    def _getMxbaiEmbedModel() -> BaseEmbedding:
        """Return Ollama embedding for mxbai-embed-large."""
        return OllamaEmbedding(
            model_name="mxbai-embed-large",
            base_url="http://localhost:11434",
        )

    @staticmethod
    def _getSnowflakeEmbedModel() -> BaseEmbedding:
        """Return Ollama embedding for snowflake-arctic-embed2."""
        return OllamaEmbedding(
            model_name="snowflake-arctic-embed2",
            base_url="http://localhost:11434",
        )