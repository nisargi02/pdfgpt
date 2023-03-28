from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os

# Load Data
#loader = UnstructuredFileLoader("state_of_the_union.txt")

def embed_doc(filename):
    if len(os.listdir("."))>0:
        loader=UnstructuredFileLoader(filename)
        raw_documents = loader.load()
        print(len(raw_documents))

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len

        )
        print("111")
        documents = text_splitter.split_documents(raw_documents)


        # Load Data to vectorstore
        embeddings = OpenAIEmbeddings()
        print("222")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("333")


        # Save vectorstore
        with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)

if os.path.exists("vectorstore.pkl"):
    with open("vectorstore.pkl","rb") as f:
        docsearch=pickle.load(f)