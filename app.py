import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

# -> BaseChatMessageHistory (Abstract Base Class)
# Abstract class: Defines the required interface for any chat history implementation.

# Cannot be used directly: You must subclass and implement its methods.

# -> ChatMessageHistory (Concrete Implementation)
# In-memory implementation of BaseChatMessageHistory.

# Stores messages in a simple Python list.


import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OllamaEmbeddings(model="llama3.2")

# Streamlit App:
st.title("Conversational RAG with PDF Uploads and chat History")
st.write("Upload PDF : ")

api_key = st.text_input("Enter your Groq API Key ",type="password")

if api_key:
    llm = ChatGroq(api_key=api_key,model="gemma2-9b-it")

    # Chat interface
    session_id = st.text_input("Session ID ",value="deafult_session")

    if 'store' not in st.session_state:
        st.session_state.store ={}

    uploaded_PDFs = st.file_uploader("Choose PDF file ",type="pdf",accept_multiple_files=True)

    #Process Uploaded file :
    if uploaded_PDFs :
        documents = []
        for uploaded_PDF in uploaded_PDFs:
            #Whenever a pdf file is uploaded,everything is in form of memory
            # So ,Creating temporary PDF file in our local 
            tempPDF = f"./temp.pdf"
            with open(tempPDF,"wb") as file:
                file.write(uploaded_PDF.getvalue())
                file_name = uploaded_PDF.name

            loader = PyPDFLoader(tempPDF)
            docs = loader.load()
            documents.extend(docs)
        

        text_splitters = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=250)
        splits = text_splitters.split_documents(documents)
        vectorstores = FAISS.from_documents(splits,embeddings)
        retriever = vectorstores.as_retriever()
    
        contextualize_q_system_prompt = (
            "Given a chat history and latest user question "
            "which might reference context in chat history , "
            "formulate a standalone question which can be understood"
            "without the chat history .Do not answer the question,"
            "just reformulate it if needed and otherwise return it as it is"
        )

        # We incorporate a variable named “chat_history” within our prompt structure, which acts as a placeholder for historical messages. By using the “chat_history” input key, we can seamlessly inject a list of previous messages into the prompt. These messages are strategically positioned after the system’s response and before the latest question posed by the user, ensuring that the context is maintained.

        contextualise_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}"),
            ]
        )


        # we employ a specialized helper function called create_history_aware_retriever. This function is crucial for managing situations where the chat history might be empty. If history is present, it constructs a sequence that effectively combines the prompt, a large language model (LLM), and a structured output parser (StrOutputParser), followed by a retriever. This sequence ensures that the latest question is contextualized within the accumulated historical data.

        # The create_history_aware_retriever function is designed to accept keys for both the input and the "chat_history", ensuring compatibility with the output schema of a standard retriever. This approach not only maintains the integrity of the interaction but also enhances the relevance and accuracy of the system’s responses.
            

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualise_q_prompt)

        # Answer question prompt : 

        systemPrompt = (
            "You are an assistant for question asnwering tasks."
            "use the following pieces of retrieved context to answer"
            "the question.If you dont know the answer , say that you"
            "dont no but no hallucination ."
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",systemPrompt),
                MessagesPlaceholder("chat_history"),
                ("user","{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id :str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question : ")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )

            st.write(st.session_state.store)
            st.write("Assistant : ",response['answer'])
            st.write("Chat History : ",session_history.messages)

else :
    st.warning("Please enter Correct Groq API key")

 