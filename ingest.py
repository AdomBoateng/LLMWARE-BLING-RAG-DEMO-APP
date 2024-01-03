###IMPORT NEEDED LIBRARIES##
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

##LOAD EMBEDDINGS TO TRANSLATE THE SENTENCES READ FROM HUMAN READABLE FORMAT TO NUMERICAL CODE FOR THE MACHINE###
embeddings = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")

##PRINT EMBEDDINGS##
print(embeddings)

###PREPARE AND LOAD PDF FILES FROM GIVEN DIRECTORY###
loader = DirectoryLoader('Data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)

##LOADED DOCUMENTS STORED IN THE 'documents' VARIABLE####
documents = loader.load()

###RESPONSE ARE GENERATED IN MANAGEABLE CHUNKS WITH GIVEN CHUNK_OVERLAP SIZE###
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)

###SPLITTED TEXTS IN DOCUMENTS STORED IN 'texts'#####
texts = text_splitter.split_documents(documents)


###CREATE VECTOR STORE USING CHROMADB####
vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space":"cosine"}, persist_directory="stores/insurance_cosine")

###PRINT STATEMENT AFTER COMPLETION###
print("Vector DB Successfully Created")