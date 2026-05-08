from dotenv import load_dotenv, find_dotenv
import logging
import os
from enum import Enum, auto
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain_openai import AzureOpenAI as LCAzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

log = logging.getLogger(__name__)

class Modeltypes(Enum):
    AZURE = 1
    OPENAI = 2
    OLLAMA = 3
    LC_AZURE = 4 #langchain version
    MXBAI = 5
    NOMIC = 6
    SAE_M_L = 7
    SAE_L = 8
    
class ModelFactory:
    """returns specific model and embedding, for future flexibility
        

    """
    @staticmethod
    def getLlmModel(modeltype):
        
        if modeltype == Modeltypes.AZURE:
            return ModelFactory._getAzureLlmModel()
        elif modeltype == Modeltypes.OLLAMA:
            return ModelFactory._getOllamaLlmModel()
        elif modeltype == Modeltypes.LC_AZURE:
            return ModelFactory._getLCAzureLlmModel()
            
    def getEmbedModel(modeltype):
        
        if modeltype == Modeltypes.AZURE:
            return ModelFactory._getAzureEmbedModel()
        if modeltype == Modeltypes.MXBAI:
            return ModelFactory._getMxbaiEmbedModel()
        if modeltype == Modeltypes.NOMIC:
            return ModelFactory._getNomicEmbedModel()
        if modeltype == Modeltypes.SAE_M_L:
            return ModelFactory._getSAE_M_L_Model()
        if modeltype == Modeltypes.SAE_L:
            return ModelFactory._getSAE_L_Model()
            
    @staticmethod
    def _getAzureLlmModel():
        """Returns AzureOpenAI Model and Embedding
           it expects the variable AZURE_OPENAI_API_KEY
           to be provided through a .env file.

        Returns:
            tupel(AzureOpenAI, AzureOpenAIEmbedding): tupel of both AzureOpenAI model and AzureOpenAI embedding
        """
        load_dotenv(find_dotenv())
        llm = AzureOpenAI(
            deployment_name="GPT-4-32k-Bot",
            api_version="2023-12-01-preview",
            model_name="gpt-4-32k",
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
        
        return llm
    
    @staticmethod
    def _getLCAzureLlmModel():
        """Returns AzureOpenAI Model and Embedding
           it expects the variable AZURE_OPENAI_API_KEY
           to be provided through a .env file.

        Returns:
            tupel(AzureOpenAI, AzureOpenAIEmbedding): tupel of both AzureOpenAI model and AzureOpenAI embedding
        """
        load_dotenv(find_dotenv())
        llm = LCAzureOpenAI(
            temperature=0,
            deployment_name="GPT-4-32k-Bot",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version="2023-12-01-preview",
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            streaming=True,
        )
        
        return llm
    
    @staticmethod
    def _getOllamaLlmModel():
        """Returns Ollama Model

        Returns:
            Ollama LLM model
        """
        llm = Ollama(model="llama3", base_url="http://10.62.197.159:11434", request_timeout=60.0)

        
        return llm
    
    @staticmethod
    def _getAzureEmbedModel():
        """Returns AzureOpenAI Model and Embedding
           it expects the variable AZURE_OPENAI_API_KEY
           to be provided through a .env file.

        Returns:
            tupel(AzureOpenAI, AzureOpenAIEmbedding): tupel of both AzureOpenAI model and AzureOpenAI embedding
        """
        load_dotenv(find_dotenv())

        embed_model = AzureOpenAIEmbedding(
            azure_endpoint="https://il-openai-pilot-gpt4.openai.azure.com/",
            model="text-embedding-ada-002",
            deployment_name="Text-embedding",
            api_version="2023-12-01-preview",
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
        
        return embed_model
    
    @staticmethod
    def _getMxbaiEmbedModel():
        """Returns Ollama Embedding

        Returns:
            OllamaEmbedding
        """
        embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1")
        
        return embed_model
    
    @staticmethod
    def _getNomicEmbedModel():
        """Returns Ollama Embedding for nomic-embed-text-v1.5 model

        Returns:
            OllamaEmbedding
        """
        embed_model = OllamaEmbedding(
            model_name="nomic_et_v1_5",
            base_url="http://10.62.197.50:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
        
        return embed_model
    @staticmethod
    def _getSAE_M_L_Model():
        """Returns Ollama Embedding for snowflake-arctic-embed:m-long model

        Returns:
            OllamaEmbedding
        """
        embed_model = OllamaEmbedding(
            model_name="gguf_sae_m_l_f16", 
            base_url="http://10.62.197.50:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
        
        return embed_model
    
    @staticmethod
    def _getSAE_L_Model():
        """Returns Ollama Embedding for snowflake-arctic-embed-l model

        Returns:
            OllamaEmbedding
        """
        """embed_model = OllamaEmbedding(
            model_name="gguf_sae_l_f16", 
            base_url="http://10.62.197.50:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )"""
        embed_model = HuggingFaceEmbedding(model_name="Snowflake/snowflake-arctic-embed-l")
        
        return embed_model
