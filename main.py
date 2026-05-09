import streamlit as st
import json
import sys, os
from dotenv import load_dotenv, find_dotenv
#sys.path.append('../')
from src.models.model_factory import Modeltypes, ModelFactory
from llama_index.core import VectorStoreIndex 
from llama_index.core import Settings
from src.database import PostgresStore
from typing import List
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
import llama_index.core.instrumentation as instrument

import logging
from src.utils import CitationFormatter

load_dotenv(find_dotenv())

logging.config.fileConfig('/home/rgraefe/git/rag_chat/log.conf')

LOAD_DATA = os.environ.get("LLAMA_LOAD_DATA", True)
DATA_DIR = os.environ.get("LLAMA_DATA_DIR", "data/")

SUM_PATH = os.environ.get("SUM_PATH", "save/index.json")

usecases_pulldown = [
    #"Agent-per-page",
    "SubQuestionQuery",
    "SubQuestionQuerywithHyde",
    "MultiStepQueryEngine",
    "MultiStepQueryEnginewithHyde"
]

@st.cache_resource
def setup_usecases():
    usecases = []
    # useCaseElem = UseCaseElement()
    # useCaseElem.usecase = Usecasetypes.MULTIDOC_LOWLEV
    # useCaseElem.nodetype = NodeTypes.SENTENCE_SPLIT
    # useCaseElem.name = "MultiDocLowLevel"
    # usecases.append(useCaseElem)

    useCaseElem = UseCaseElement()
    useCaseElem.usecase = Usecasetypes.MULTI_QUERY
    useCaseElem.nodetype = NodeTypes.WINDOW_SPLIT
    useCaseElem.name = "MultiQuery"
    usecases.append(useCaseElem)

    useCaseElem = UseCaseElement()
    useCaseElem.usecase = Usecasetypes.MULTI_QUERY_HYDE
    useCaseElem.nodetype = NodeTypes.WINDOW_SPLIT
    useCaseElem.name = "MultiQueryHyde"
    usecases.append(useCaseElem)

    useCaseElem = UseCaseElement()
    useCaseElem.usecase = Usecasetypes.SUB_QUERY
    useCaseElem.nodetype = NodeTypes.WINDOW_SPLIT
    useCaseElem.name = "SubQuery"
    usecases.append(useCaseElem)

    useCaseElem = UseCaseElement()
    useCaseElem.usecase = Usecasetypes.SUB_QUERY_HYDE
    useCaseElem.nodetype = NodeTypes.WINDOW_SPLIT
    useCaseElem.name = "SubQueryHyde"
    usecases.append(useCaseElem)
    
    
    storage = PostgresStore("postgresql://admin:admin@127.0.0.1:5433/vectordb", "vector_db")

    #docs = read_documents(args=args)
    docstore = storage.get_doc_store("d_parent")
    docs = list(docstore.docs.values())
    _agents = {}

    for elem in usecases:
        _agent = {}
        log_event_handler = RetrieverEventHandler()
        event_handler=IntermediateEventHandler()
        span_handler = SimpleSpanHandler()
        dispatcher = instrument.get_dispatcher()  
        dispatcher.event_handlers=[]
        dispatcher.add_event_handler(event_handler)
        dispatcher.add_event_handler(log_event_handler)
        dispatcher.add_span_handler(span_handler)
        nodetype = elem.nodetype
        usecasetype = elem.usecase
        name = elem.name
        agent = UseCaseFactory.getUsecase(usecasetype=usecasetype, nodetype=nodetype, docs=docs,sum_path=SUM_PATH,cached_documents=True)
        _agent["dispatcher"] = dispatcher
        _agent["agent"] = agent
        _agents[name] = _agent
    return _agents

def on_change_selectbox():
    usecase_selected = st.session_state["usecase_selectbox"]
    st.session_state.messages = [
        {"role": "assistant", "content": f"Ask me a question about Intels Automotive platforms! using {usecase_selected}"}
    ]
    
    if usecase_selected ==  "Agent-per-page":
        st.session_state.chat_engine = st.session_state["agents"]["MultiDocLowLevel"]["agent"]
        st.session_state.dispatcher = st.session_state["agents"]["MultiDocLowLevel"]["dispatcher"]
    elif usecase_selected ==  "SubQuestionQuery":
        st.session_state.chat_engine = st.session_state["agents"]["SubQuery"]["agent"]
        st.session_state.dispatcher = st.session_state["agents"]["SubQuery"]["dispatcher"]
    elif usecase_selected ==  "SubQuestionQuerywithHyde":
        st.session_state.chat_engine = st.session_state["agents"]["SubQueryHyde"]["agent"]
        st.session_state.dispatcher = st.session_state["agents"]["SubQueryHyde"]["dispatcher"]
    elif usecase_selected ==  "MultiStepQueryEngine":
        st.session_state.chat_engine = st.session_state["agents"]["MultiQuery"]["agent"]
        st.session_state.dispatcher = st.session_state["agents"]["MultiQuery"]["dispatcher"]
    elif usecase_selected ==  "MultiStepQueryEnginewithHyde":
        st.session_state.chat_engine = st.session_state["agents"]["MultiQueryHyde"]["agent"]
        st.session_state.dispatcher = st.session_state["agents"]["MultiQueryHyde"]["dispatcher"]

st.session_state["agents"] = setup_usecases()

with st.sidebar:
    if 'usecase_selectbox' not in st.session_state:
        st.session_state['usecase_selectbox'] = usecases_pulldown[0]
    
    usecase_selected = st.selectbox('Choose the use case:', usecases_pulldown, on_change=on_change_selectbox,
                                    key="usecase_selectbox")
    st.markdown("""
**Usecases:**  
1. Agent-per-page:  
an agent is assigned to each document section or page if the document does not have sections,
Agents have a summary for their document chunk assigned. Based on that they are selected
according to their similarity to the query. SubQueries are generated and assigned to different
Agents. The LLM answeres each sub-query individually and generates a final answer based on its
intermediate answers.  

2. SubQuestionQuery:  
Document chunks are split using a window method where for each sentence the encodings are stored
in a vectordatabase together with a link to the sentence before and after. All 3 sentences are
send to the LLM. All windowed chunks are accessed using the same query engine. On top a 
SubQuestionQuery engine generates multiple questions which are subsequently answered by the LLM.

3. SubQuestionQuerywithHyde  
same as 2. with the addition of Hyde method. Hyde takes the original query and asks the LLM to 
generate a longer text paragraph based on that. This paragraph instead of the original query 
is used for embeddings lookup.

4. MultiStepQueryEngine:  
In contrast to the SubQuestionQuery the answer to a previous question is presented to the LLM 
for additional context. However there is no final request that presents all sub-queries together to the LLM.

5. MultiStepQueryEnginewithHyde:  
same as 4. with Hyde. See 3. for a description of Hyde.
                """)


if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": f"Ask me a question about Intels Automotive platforms! using {usecase_selected}"}
    ]
    
if usecase_selected ==  "Agent-per-page":
    st.session_state.chat_engine = st.session_state["agents"]["MultiDocLowLevel"]["agent"]
    st.session_state.dispatcher = st.session_state["agents"]["MultiDocLowLevel"]["dispatcher"]
elif usecase_selected ==  "SubQuestionQuery":
    st.session_state.chat_engine = st.session_state["agents"]["SubQuery"]["agent"]
    st.session_state.dispatcher = st.session_state["agents"]["SubQuery"]["dispatcher"]
elif usecase_selected ==  "SubQuestionQuerywithHyde":
    st.session_state.chat_engine = st.session_state["agents"]["SubQueryHyde"]["agent"]
    st.session_state.dispatcher = st.session_state["agents"]["SubQueryHyde"]["dispatcher"]
elif usecase_selected ==  "MultiStepQueryEngine":
    st.session_state.chat_engine = st.session_state["agents"]["MultiQuery"]["agent"]
    st.session_state.dispatcher = st.session_state["agents"]["MultiQuery"]["dispatcher"]
elif usecase_selected ==  "MultiStepQueryEnginewithHyde":
    st.session_state.chat_engine = st.session_state["agents"]["MultiQueryHyde"]["agent"]
    st.session_state.dispatcher = st.session_state["agents"]["MultiQueryHyde"]["dispatcher"]

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.html(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking... Due to multiple backend requests this can take several minutes."):
            response = st.session_state.chat_engine.query(prompt)
            dispatcher = st.session_state.dispatcher
            history = dispatcher.event_handlers[0].query_history
            merged_response = CitationFormatter.merge_cite_results(history=history)
            st.html(merged_response)
            message = {"role": "assistant", "content": merged_response}
            st.session_state.messages.append(message) # Add response to message history
