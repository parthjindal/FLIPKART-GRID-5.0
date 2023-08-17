import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY
from query_refiner import QueryRefiner
from main2 import get_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state["buffer_memory"] = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
memory = st.session_state["buffer_memory"]
queryRefiner = QueryRefiner(llm,memory)
agent = get_agent(llm,memory)

def get_response(query):
    query = queryRefiner(query)
    response = agent.run(input = query)
    return response

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            st.subheader("Refined Query:")
            st.write(query)
            response = get_response(query)
            response += "\nhttps://www.flipkart.com/nova-nht-1039-usb-trimmer-45-min-runtime-4-length-settings/p/itmfd97844846452?pid=TMRGB9NKKVZTGTK9&lid=LSTTMRGB9NKKVZTGTK99MO9J5&marketplace=FLIPKART&store=zlw&srno=b_1_2&otracker=hp_omu_Best%2Bof%2BElectronics_1_4.dealCard.OMU_W2LWSC2NR71R_3&otracker1=hp_omu_PINNED_neo%2Fmerchandising_Best%2Bof%2BElectronics_NA_dealCard_cc_1_NA_view-all_3&fm=neo%2Fmerchandising&iid=en_9poLyJmV7zEI1xvNKf0jC5dCEiioyraoOc6l5RMnTePBeM2YuweVmWDI2jzFcYMAtyzm4mNqPa0l76DWQVFZyw%3D%3D&ppt=browse&ppn=browse"
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


