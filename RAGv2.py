import gradio as gr
import ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
import numpy as np

loader = DirectoryLoader('/home/giu/PFC2/LLMs-for-improving-Big-Five-classification/HOPE_WSDM_2022/All', glob="**/*.md", show_progress=True, use_multithreading=True) # loader_cls=CSVLoader
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

try:
    embeddings = OllamaEmbeddings(model="llama3")
except ValueError as e:
    print("Error connecting to Ollama server:", e)
    # Si hay un error, usa embeddings de HuggingFace
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    embeddings = embed_model

llm = Ollama(model="llama3", request_timeout=120.0)

vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)


Settings.embed_model = embeddings
Settings.llm = llm
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)

qa_prompt_tmpl_str = (
    "You are a psychotherapist engaging in a conversation with a client. "
    "Your goal is to ask insightful and probing questions to understand the client's personality better. "
    "Use the context provided to guide your questions, ensuring they are reflective, empathetic, and designed to encourage deep introspection. "
    "If the context doesn't provide enough information, ask open-ended questions to elicit more detailed responses. "
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above, formulate your response in the form of a question that a psychotherapist would ask. "
    "Your questions should help uncover the client's underlying thoughts, feelings, and motivations.\n"
    "Query: {query_str}\n"
    "Response: "
)

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})

response = query_engine.query('What exactly is DSPy?') # Aquí el usuario interactúa con el modelo, esto es ejemplo
print(response)
