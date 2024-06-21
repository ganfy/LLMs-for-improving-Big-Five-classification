from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import torch

# Load documents
documents = DirectoryLoader("/home/giu/PFC2/LLMs-for-improving-Big-Five-classification/HOPE_WSDM_2022/All").load()

# Setup embedding model
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# System prompt for the LLM
system_prompt = (
    "You are a psychotherapist engaging in a conversation with a client. "
    "Your goal is to ask insightful and probing questions to understand the client's personality better. "
    "Use the example provided to guide your questions, ensuring they are reflective, empathetic, and designed to encourage deep introspection. "
    "If the client doesn't provide enough information, ask open-ended questions to elicit more detailed responses. "
    "Your questions should help uncover the client's underlying thoughts, feelings, and motivations.\n"
    "Once you've gathered enough information from the client in the areas: openness, conscientiousness, extraversion, agreeableness, and neuroticism, respond with the word: 'Enough'."
)
query_wrapper_prompt = PromptTemplate(template="User input: {query_str}\nRetrieved example: ", input_variables=["query_str"])

# Setup LLM with GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm = Ollama(model="llama3", system=system_prompt)

# Setup Qdrant client and vector store
client = QdrantClient(":memory:", port=6333, grpc_port=6333)
vector_store = Qdrant(client=client, collection_name="test", embeddings=embed_model)
vector_store.from_documents(documents, embedding=embed_model)

# Rerank setup
rerank_model = SentenceTransformer("cross-encoder/ms-marco-MiniLM-L-2-v2")
rerank = lambda x: rerank_model.encode(x, convert_to_tensor=True)

# Setup RetrievalQA chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain(llm, retriever, post_processor=rerank)

# Function to truncate the conversation history to fit within token limits
def truncate_conversation(conversation, max_tokens=2048):
    tokens = []
    truncated_conversation = []
    for entry in reversed(conversation):
        entry_tokens = entry.split()
        if len(tokens) + len(entry_tokens) + 50 > max_tokens:  # 50 tokens buffer for prompt parts
            break
        tokens = entry_tokens + tokens
        truncated_conversation.insert(0, entry)
    return truncated_conversation

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
    formatted_query = f"Conversation history: {short_term_memory}\nUser input: {user_input}\nRetrieved context: "

    response = qa_chain(formatted_query)
    response_text = str(response["output_text"])

    if "Enough" in response_text:
        sufficient_info = True
    else:
        print(f"Psychotherapist: {response_text}")
        conversation.append(f"Psychotherapist: {response_text}")

conversation_text = "\n".join(conversation)

print("Psychotherapist: Thanks for sharing. That's all for today.")

with open("conversation.txt", "w") as file:
    file.write(conversation_text)
