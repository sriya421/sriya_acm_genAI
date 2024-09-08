import os
import json

import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate



working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data["GROQ_API_KEY"]

# save the api key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# streamlit page configuration
st.set_page_config(
    page_title="My Chatbot",
    page_icon="🦙",
    layout="centered"
)

# Add customization options to the sidebar
st.sidebar.title('Select an LLM')

model2 = st.sidebar.selectbox(
    'Choose a model',
    ['mixtral-8x7b-32768', 'llama3-70b-8192', 'gemma2-9b-it']
)

conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

memory=ConversationBufferWindowMemory(k=conversational_memory_length)


client = Groq()

# initialize the chat history as streamlit session state of not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# streamlit page title
st.title("🦙 My ChatBot")

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# input field for user's message:
user_prompt = st.chat_input(f"Ask {model2}...")

if user_prompt:

    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # sends user's message to the LLM and get a response
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        *st.session_state.chat_history
    ]

    response = client.chat.completions.create(
        #model="llama-3.1-8b-instant",
        model=model2,
        messages=messages
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

