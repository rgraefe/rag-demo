from dotenv import load_dotenv, find_dotenv
import logging
import os
from enum import Enum, auto
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_openai import AzureOpenAI as LCAzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

log = logging.getLogger(__name__)

class Modeltypes(Enum):
    OPENAI = 1
    OLLAMA = 2
    NOMIC = 3
    SNOWFLAKE = 4
    MXBAI = 5
class ModelFactory:
    """returns specific model and embedding, for future flexibility
        

    """
    @staticmethod
    def getLlmModel(modeltype: Modeltypes):
        

        if modeltype == Modeltypes.OPENAI:
            return ModelFactory._getOpenAILlmModel()
        if modeltype == Modeltypes.OLLAMA:
            return ModelFactory._getOllamaLlmModel()
            
    @staticmethod
    def getEmbedModel(modeltype: Modeltypes):
        
        if modeltype == Modeltypes.OPENAI:
            return ModelFactory._getOpenAIEmbedModel()
        if modeltype == Modeltypes.NOMIC:
            return ModelFactory._getNomicEmbedModel()
        if modeltype == Modeltypes.MXBAI:
            return ModelFactory._getMxbaiEmbedModel()
        if modeltype == Modeltypes.SNOWFLAKE:
            return ModelFactory._getSnowflakeEmbedModel() 
    
  
    @staticmethod
    def _getOpenAILlmModel():
        """Returns OpenAI Model
           it expects the variable OPENAI_API_KEY
           to be provided through a .env file.

        Returns:
            OpenAI LLM model
        """
        load_dotenv(find_dotenv())
        llm = OpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        return llm
    
    @staticmethod
    def _getOllamaLlmModel():
        """Returns Ollama Model

        Returns:
            Ollama LLM model
        """
        llm = Ollama(model="llama3.2", base_url="http://localhost:11434", request_timeout=60.0)

        
        return llm
    
    
    @staticmethod
    def _getOpenAIEmbedModel():
        """Returns OpenAI Embedding Model
           it expects the variable OPENAI_API_KEY
           to be provided through a .env file.

        Returns:
            OpenAIEmbedding: OpenAI embedding model
        """
        load_dotenv(find_dotenv())

        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        return embed_model
    
    
    @staticmethod
    def _getNomicEmbedModel():
        """Returns Ollama Embedding for nomic-embed-text-v1.5 model

        Returns:
            OllamaEmbedding
        """
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
        
        return embed_model
    
    @staticmethod
    def _getMxbaiEmbedModel():
        """Returns Ollama Embedding for mxbai-embed-large-v1 model

        Returns:
            OllamaEmbedding
        """
        embed_model = OllamaEmbedding(
            model_name="mxbai-embed-large",
            base_url="http://localhost:11434",
        )
        
        return embed_model
    
    @staticmethod
    def _getSnowflakeEmbedModel():
        """Returns Ollama Embedding for snowflake-arctic-embed2 model

        Returns:
            OllamaEmbedding
        """
        embed_model = OllamaEmbedding(
            model_name="snowflake-arctic-embed2",
            base_url="http://localhost:11434",
        )
        
        return embed_model

