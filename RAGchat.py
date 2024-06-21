import csv
import os
from llama_index.core import SimpleDirectoryReader, Settings, PromptTemplate, VectorStoreIndex, StorageContext
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama
from llama_index.core.types import BaseModel
from typing import List
import qdrant_client
import torch

# Load documents
documents = SimpleDirectoryReader("/home/giu/PFC2/LLMs-for-improving-Big-Five-classification/HOPE_WSDM_2022/All").load_data()

# Setup embedding model
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

from typing import List
from pydantic import BaseModel


class Response(BaseModel):
    """Data model for a response from a psychotherapist asking questions to the patient."""

    question_to_patient: str
    gathered_info_BigFive: List[bool]
    annotations: str

# System prompt for the LLM
system_prompt = (
    "You are a psychotherapist engaging in a conversation with a client. "
    "Your goal is to ask insightful and probing questions to understand the client's personality better. "
    "Use the retrieved examples of other interactions to guide your questions, ensuring they are reflective, empathetic, and designed to encourage deep introspection. "
    "The retrieved examples are not related to the client's input but should help you form better questions. "
    "Do not provide explanations or meta-comments. Only respond straightforward with the next question or statement you would make as a psychotherapist. "
    "If the client doesn't provide enough information, ask open-ended questions to elicit more detailed responses. "
    "Your questions should help uncover the client's underlying thoughts, feelings, and motivations, following the SOCRATIC METHOD.\n"
    "Once you've gathered enough information from the client in the areas: openness, conscientiousness, extraversion, agreeableness, and neuroticism, respond with the word: 'Enough'."
    "The conversation shouldn't be longer than 2048 tokens."
)

# Setup LLM with GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm = Ollama(model="llama3", request_timeout=300.0, system=system_prompt, device=device)

Settings.llm = llm
Settings.chunk_size = 512

# Setup Qdrant client and vector store
client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Rerank setup
rerank = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank], response_mode="tree_summarize")

# Function to truncate the conversation history to fit within token limits
def truncate_conversation(conversation, max_tokens=1024):
    tokens = []
    truncated_conversation = []
    for entry in reversed(conversation):
        entry_tokens = entry.split()
        if len(tokens) + len(entry_tokens) + 50 > max_tokens:  # 50 tokens buffer for prompt parts
            break
        tokens = entry_tokens + tokens
        truncated_conversation.insert(0, entry)
    return truncated_conversation

# Initialize CSV file
csv_file = "conversation.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Answers"])  # Write header

# Chatbot loop
conversation = []
answers = []
sufficient_info = False

initial_prompt = "Psychotherapist: Hi, how are you feeling today?"
print(initial_prompt)
conversation.append(initial_prompt)

while not sufficient_info:
    user_input = input("You: ")
    conversation.append(f"Client: {user_input}")
    answers.append(user_input)

    if user_input.lower() == "exit":
        break

    # Truncate conversation history to fit within token limits
    truncated_conversation = truncate_conversation(conversation)
    short_term_memory = "\n".join(truncated_conversation)
    formatted_query = f"Conversation history: {short_term_memory}\nUser input: {user_input}\nPsychotherapist question: "

    response = query_engine.query(formatted_query)
    response_text = str(response)

    if "Enough" in response_text:
        sufficient_info = True
    else:
        print(f"Psychotherapist: {response_text}")
        conversation.append(f"Psychotherapist: {response_text}")

conversation_text = "\n\n" + "\n".join(conversation)

print("Psychotherapist: Thanks for sharing. That's all for today.")

# Save conversation to TXT
with open("conversation.txt", "a") as file:
    file.write(conversation_text + "\n\n")

# Save answers to CSV
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(answers)
