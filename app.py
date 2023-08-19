import streamlit as st
from streamlit_chat import message
from recommender_tool import SearchCache
from langchain.chat_models import ChatOpenAI
from config import *
from query_refiner import QueryRefiner
from main import get_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
import random

st.subheader("Chat APD: Attire Predicting Dost Presented by Aryan, Parth and Divyangna")


if 'responses' not in st.session_state:
    st.session_state['responses'] = [("How can I assist you?",None)]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state["buffer_memory"] = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

if 'current_index' not in st.session_state:
    st.session_state["current_index"] = {}

if 'current_image' not in st.session_state:
    st.session_state["current_image"] = None

if 'agent' not in st.session_state:
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY3)
    memory = st.session_state["buffer_memory"]
    queryRefiner = QueryRefiner(llm,memory)
    searchCache = SearchCache()
    st.session_state["searchCache"] = searchCache
    st.session_state['agent'] = get_agent(llm,memory,searchCache)

def get_next_item_getter(images, key):
    def get_next_item():
        current_index = st.session_state["current_index"].get(key,0)
        current_index = min(current_index+1, len(images)-1)
        st.session_state["current_index"][key] = current_index
    return get_next_item

def get_previous_item_getter(images, key):
    def get_previous_item():
        current_index = st.session_state["current_index"].get(key,0)
        current_index = max(current_index-1, 0)
        st.session_state["current_index"][key] = current_index
    return get_previous_item


# st.button("Get Next Item",on_click=get_next_item)
# st.button("Get Previous Item",on_click=get_previous_item)


def get_response(query):
    # query = queryRefiner(query)
    agent = st.session_state['agent']
    searchCache = st.session_state['searchCache']
    response = agent.run(input = query)
    search_json = searchCache.getAll()
    images = []
    for item in search_json:
        print(item)
        link = item["link"]
        thumbnail = item["thumbnail"]
        images.append(f'<figure><img width="100%" height="200" src="{thumbnail}"/><figcaption>{link}</figcaption></figure>')
                      
    return f'<p style="font-family:robotica back;">{response}</p>', images

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


def funcy():
    st.session_state["query"] = st.session_state.input
    st.session_state.input = ""

with textcontainer:
    st.text_input("Query: ", key="input",on_change= funcy)
    query = st.session_state.get("query", "")
    st.session_state["query"] = ""
    if query:
        with st.spinner("typing..."):
            st.subheader("Refined Query:")
            st.write(query)
            response, images = get_response(query)
        st.session_state.requests.append(query)
        st.session_state.responses.append((response,images)) 

with response_container:
    # message(st.session_state['current_image'],key="-1",allow_html=True)
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            response, images = st.session_state['responses'][i]
            message(response,key=str(i),allow_html=True)
            if images:
                current_index = st.session_state["current_index"].get(i,0)
                message(images[current_index],key=str(i)+"_image",allow_html=True)
                col1, col2 = st.columns([3,3])
                with col1:
                    st.button("Previous",on_click=get_previous_item_getter(images,i),key=str(i)+"_previous",use_container_width=True)
                with col2:
                    st.button(" Next",on_click=get_next_item_getter(images, i),key=str(i)+"_next",use_container_width=True)

            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


