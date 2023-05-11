# chatbot.py
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

qa = None

def initialize_chatbot(config):
    global qa
    llm = ChatOpenAI(temperature=config['chatbot']['temperature'],
                     model_name=config['chatbot']['model_name'])

    loader = GitLoader(
        repo_path=config['gitloader']['repo_path'], 
        branch=config['gitloader']['branch'], 
        file_filter=lambda file_path: file_path.endswith(".md")
        )
    
    data = loader.load()
    print("Number of documents loaded:", len(data))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['text_splitter']['chunk_size'], 
        chunk_overlap=config['text_splitter']['chunk_overlap'],
        separators=[" ", ",", "\n"]
        )

    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = Chroma.from_documents(texts, 
                               embeddings, 
                               persist_directory=config['chroma']['persist_directory'])

    qa = ConversationalRetrievalChain.from_llm(
        llm, db.as_retriever(), 
        return_source_documents=True
        )
    print(f'Chatbot initialized: {qa}')
    return qa