'''
from openai import OpenAI



client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
messages = [
    {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}
]

def send_message(user_content):
    # Añadir el mensaje del usuario al historial
    global messages
    messages.append({"role": "user", "content": user_content})

    # Crear la respuesta basada en el historial de mensajes
    completion = client.chat.completions.create(
        model="local-model",
        messages=messages,  # Pasa el historial completo aquí
        temperature=0.7,
    )

    # Obtener la respuesta del asistente
    assistant_message = completion.choices[0].message.content

    # Añadir la respuesta del asistente al historial
    messages.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message

# Ejemplo de uso
print(send_message("hola como estas"))
'''


'''
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams, CreateCollection
import hashlib

# Initialize the OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# Initialize Qdrant client for vector memory
qdrant = QdrantClient(host="localhost", port=6333)

# Ensure the collection exists in Qdrant
vector_params = VectorParams(size=512, distance="Cosine")
collection_name = "chat_history"
qdrant.recreate_collection(
    collection_name,
    vectors_config={"default": vector_params},  # Add this line to specify the vector configuration
)
# qdrant.recreate_collection(collection_name, vector_size=512, distance="Cosine")

# Function to generate vector from text (for simplicity, let's assume a dummy vector)
def text_to_vector(text):
    # In a real scenario, use an embedding model to convert text to a vector
    return [hashlib.sha256(text.encode()).hexdigest()] * 512

def store_message(role, content):
    vector = text_to_vector(content)
    point_id = hashlib.sha256(f"{role}_{content}".encode()).hexdigest()
    point = PointStruct(id=point_id, vector=vector, payload={"role": role, "content": content})
    qdrant.upsert(collection_name, [point])

def retrieve_messages():
    points = qdrant.scroll(collection_name, limit=100)
    messages = [{"role": point.payload["role"], "content": point.payload["content"]} for point in points]
    return messages

def send_message(user_content):
    # Store the user's message
    store_message("user", user_content)

    # Retrieve the conversation history
    messages = retrieve_messages()

    # Create the response based on the retrieved history
    completion = client.chat.completions.create(
        model="local-model",
        messages=messages,  # Pass the entire conversation history here
        temperature=0.7,
    )

    # Get the assistant's response
    assistant_message = completion.choices[0].message.content

    # Store the assistant's response
    store_message("assistant", assistant_message)
    
    return assistant_message

# Example usage
print(send_message("hola como estas"))
'''
# # # # # # VERSIÓN FAISS

# from openai import OpenAI
# from langchain.vectorstores import FAISS
# from langchain.memory import VectorStoreRetrieverMemory
# from langchain.chains import LLMChain
# from langchain.prompts import ChatPromptTemplate
# import faiss
# import numpy as np
# from langchain_community.docstore.in_memory import InMemoryDocstore

# # Inicializa el cliente con tu modelo local
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
# messages = [
#     {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}
# ]

# from fastembed.embedding import DefaultEmbedding

# embedder = DefaultEmbedding()

# def custom_embed(text):
#     embedding = list(embedder.embed(text))  # Convierte a lista o numpy array
#     return np.array(embedding)

# # Inicializa el índice FAISS
# initial_embedding = custom_embed("test")
# embedding_dim = initial_embedding.shape[0]
# index = faiss.IndexFlatL2(embedding_dim)

# # Inicializa el docstore como un diccionario para almacenar documentos
# docstore = {}
# index_to_docstore_id = []

# vectorstore = FAISS(
#     embedding_function=custom_embed,
#     index=index,
#     docstore= InMemoryDocstore(),
#     index_to_docstore_id=index_to_docstore_id
# )

# memory = VectorStoreRetrieverMemory(
#     retriever=vectorstore.as_retriever(k=1),
#     memory_key="history",
# )

# # Define el template del prompt
# prompt_template = ChatPromptTemplate.from_template(
#     "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados.\n\n"
#     "Relevant pieces of previous conversation:\n{history}\n\n"
#     "(You do not need to use these pieces of información if not relevant)\n\n"
#     "Current conversation:\nHuman: {input}\nAI:"
# )

# def llm_function(prompt):
#     return client.chat.completions.create(
#         model="local-model",
#         messages=messages + [{"role": "user", "content": prompt}],
#         temperature=0.7
#     ).choices[0].message.content

# # Crear una instancia de RunnableLambda
# from langchain_core.runnables import RunnableLambda
# llm = RunnableLambda(llm_function)

# # Crear la cadena LLMChain
# chain = LLMChain(
#     llm=llm,
#     prompt=prompt_template,
#     memory=memory
# )

# # Función para agregar documentos al docstore
# def add_document_to_memory(user_content, assistant_message):
#     # Genera el embedding del mensaje del usuario
#     embedding = custom_embed(user_content)

#     # Agrega el documento al índice y al docstore
#     index.add(np.array([embedding]))  # Asegúrate de que el formato sea correcto
#     doc_id = len(docstore)  # Genera un ID único
#     docstore[doc_id] = user_content  # Almacena el documento
#     index_to_docstore_id.append(doc_id)  # Mantén un seguimiento de los IDs

# # Función para enviar un mensaje con integración de memoria
# def send_message(user_content):
#     global messages
#     messages.append({"role": "user", "content": user_content})

#     # Guarda el contexto actual en la memoria
#     memory.save_context({"input": user_content}, {"output": None})

#     # Genera la respuesta usando la cadena con memoria
#     assistant_message = chain.run({"input": user_content})

#     # Almacena la respuesta del asistente en messages
#     messages.append({"role": "assistant", "content": assistant_message})

#     # Agrega el documento a la memoria
#     add_document_to_memory(user_content, assistant_message)

#     return assistant_message

# # Ejemplo de uso
# print(send_message("hola, ¿cómo estás?"))


# # # # # VERSIÓN QDRANT + VECTORSTORERETRIEVERMEMORY
from openai import OpenAI
from langchain.vectorstores import Qdrant
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from fastembed.embedding import DefaultEmbedding
from qdrant_client import QdrantClient

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
messages = [
    {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}
]

embedder = DefaultEmbedding()

def custom_embed(text):
    return list(embedder.embed(text))  # Convierte a lista o numpy array

def check_qdrant():
    try:
        response = requests.get("http://localhost:6333/dashboard/")
        return response.status_code == 200
    except requests.ConnectionError:
        return False
    
if check_qdrant():
    client = QdrantClient(url="http://localhost:6333")
    print("Connected to local Qdrant server.")
    
vectorstore = Qdrant(
    client=qdrant_client,
    embedding_function=custom_embed,
    collection_name="my_collection",
)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(k=1),
    memory_key="history",
)

system_prompt = ('''
You are a psychotherapist conducting a conversation with a client. Your main goals are to gather information about the client's personality and avoid dataset bias. 
Use the retrieved examples to help form your questions, ensuring they are reflective, empathetic, and encourage deep introspection.

1. Formulate your responses as insightful questions or statements to guide the client's self-reflection.
2. Avoid repeating questions from the conversation history.
3. Ensure your questions are designed to elicit detailed and unbiased responses.
4. Use the Socratic method to explore the client's thoughts, feelings, and motivations.
5. If the client provides limited information, ask open-ended questions to gather more details.
6. Avoid using excessive negative language unless there are clear indicators of neuroticism. Strive to use neutral or positive wording where appropriate to avoid misinterpreting the client's statements.
7. Your goal is to collect sufficient information in the areas of openness, conscientiousness, extraversion, agreeableness, and neuroticism.

Only respond with the next question or statement you would make as a psychotherapist. Once you have gathered enough information in all five areas, respond with 'Enough'. Ensure the conversation does not exceed 2048 tokens. The client may respond in English or Spanish.
'''
)

prompt_template = ChatPromptTemplate.from_template(
    system_prompt + "\n"
    "Relevant pieces of previous conversation:\n{history}\n\n"
    "(You do not need to use these pieces of información if not relevant)\n\n"
    "Current conversation:\nHuman: {input}\nAI:"
)

def llm_function(prompt):
    return client.chat.completions.create(
        model="local-model",
        messages=messages + [{"role": "user", "content": prompt}],
        temperature=0.7
    ).choices[0].message.content


from langchain_core.runnables import RunnableLambda
llm = RunnableLambda(llm_function)


chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory
)


def add_document_to_memory(user_content, assistant_message):
    embedding = custom_embed(user_content)

    vectorstore.add_texts([user_content], embeddings=[embedding])

# Función para enviar un mensaje con integración de memoria
def send_message(user_content):
    global messages
    messages.append({"role": "user", "content": user_content})

    memory.save_context({"input": user_content}, {"output": None})

    assistant_message = chain.run({"input": user_content})

    messages.append({"role": "assistant", "content": assistant_message})

    add_document_to_memory(user_content, assistant_message)

    return assistant_message

# Ejemplo de uso
print(send_message("hola, ¿cómo estás?"))

