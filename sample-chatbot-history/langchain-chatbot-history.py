import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

model= ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

from langchain_core.messages import HumanMessage
model.invoke([HumanMessage(content="I love dogs as a pet animal. Please tell me more about dogs.")])

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return ChatMessageHistory()

with_message_history=RunnableWithMessageHistory(model, get_session_history)

config1 = {"configurable": {"session_id": "chatbot-session1"}}

with_message_history.invoke(
    [HumanMessage(content="I love dogs as a pet animal. Please tell me more about dogs.")],
    config=config1)

with_message_history.invoke(
    [HumanMessage(content="What pet animal do I like?")],
    config=config1)

# change the config
config2 = {"configurable": {"session_id": "chatbot-session2"}}

with_message_history.invoke(
    [HumanMessage(content="What pet animal do I like?")],
    config=config2)