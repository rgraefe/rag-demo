"""Default prompt selectors."""
from llama_index.core.prompts import SelectorPromptTemplate
from .cite_prompts import (
    CITATION_QA_PROMPT,
    CITATION_CHAT_QA_PROMPT,
    CITATION_REFINE_PROMPT,
    CITATION_CHAT_REFINE_PROMPT
)

from llama_index.core.prompts.utils import is_chat_model


COHERE_QA_TEMPLATE = None
COHERE_REFINE_TEMPLATE = None
COHERE_TREE_SUMMARIZE_TEMPLATE = None
COHERE_REFINE_TABLE_CONTEXT_PROMPT = None

# Define prompt selectors for Text QA, Tree Summarize, Refine, and Refine Table.
# Note: Cohere models accept a special argument `documents` for RAG calls. To pass on retrieved documents to the `documents` argument,
# specialised templates have been defined. The conditionals below ensure that these templates are called by default when a retriever
# is called with a Cohere model for generator.

# Text QA
citation_text_qa_conditionals = [(is_chat_model, CITATION_CHAT_QA_PROMPT)]

CITATION_TEXT_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=CITATION_QA_PROMPT,
    conditionals=citation_text_qa_conditionals,
)

# Refine
citation_refine_conditionals = [(is_chat_model, CITATION_CHAT_REFINE_PROMPT)]

CITATION_REFINE_PROMPT_SEL = SelectorPromptTemplate(
    default_template=CITATION_REFINE_PROMPT,
    conditionals=citation_refine_conditionals,
)

