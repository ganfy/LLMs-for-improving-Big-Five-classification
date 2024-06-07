from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings, PromptTemplate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM 
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
import time
from llama_index.llms.ollama import Ollama
import qdrant_client


documents = SimpleDirectoryReader("/home/giu/PFC2/LLMs-for-improving-Big-Five-classification/HOPE_WSDM_2022/All").load_data()

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

system_prompt = (
    "You are a psychotherapist engaging in a conversation with a client. "
    "Your goal is to ask insightful and probing questions to understand the client's personality better. "
    "Use the example provided to guide your questions, ensuring they are reflective, empathetic, and designed to encourage deep introspection. "
    "If the client doesn't provide enough information, ask open-ended questions to elicit more detailed responses. "
    "Your questions should help uncover the client's underlying thoughts, feelings, and motivations.\n"
    "Once you've gathered enough information from the client in the areas: openness, conscientiousness, extraversion, agreeableness, and neuroticism, respond wiht the word: 'Enough'."
)
query_wrapper_prompt = PromptTemplate("{query_str}")

llm = Ollama(model="llama3", request_timeout=120.0)

Settings.llm = llm
Settings.chunk_size = 512

client = qdrant_client.QdrantClient(location=":memory:")

vector_store = QdrantVectorStore(client=client, collection_name="test")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Rerank
rerank = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank] )


# Chatbot loop
conversation = []
sufficient_info = False

print("Psychotherapist: Hi, how are you feeling today?")

while not sufficient_info:
    user_input = input("You: ")
    conversation.append(f"You: {user_input}")
    
    response = query_engine.query(user_input)
    response_text = str(response)
    
    if "Enough" in response_text:
        sufficient_info = True
    else:
        print(f"Psychotherapist: {response_text}")
        conversation.append(f"Psychotherapist: {response_text}")

conversation_text = "\n".join(conversation)
with open("conversation.txt", "w") as file:
    file.write(conversation_text)

print("Psychotherapist: Thanks for sharing. That's all for today.")