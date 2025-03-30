import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

def get_vectorstore_from_url(url):
    try:
        # Load content from the URL
        loader = WebBaseLoader(url)
        document = loader.load()
        
        # Split the document into chunks with proper configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(document)
        
        # Create vector store using Gemini embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        vector_store = Chroma.from_documents(
            document_chunks,
            GoogleGenerativeAIEmbeddings(api_key=api_key, model="models/embedding-001")
        )
        return vector_store
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_context_retriever_chain(vector_store):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")  # Updated model name

        retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up information relevant to the conversation.")
        ])

        retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
        return retriever_chain
    
    except Exception as e:
        st.error(f"Error creating retriever chain: {str(e)}")
        return None
    
def get_conversational_rag_chain(retriever_chain):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")  # Updated model name

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def get_response(user_input):
    try:
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        if not retriever_chain:
            return "Error initializing retriever chain"
        
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        if not conversation_rag_chain:
            return "Error initializing conversation chain"
        
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        
        return response['answer']
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# App configuration
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if not website_url:
    st.info("Please enter a website URL")
else:
    # Initialize or update vector store when URL changes
    if "website_url" not in st.session_state or st.session_state.website_url != website_url:
        st.session_state.website_url = website_url
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    
    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
