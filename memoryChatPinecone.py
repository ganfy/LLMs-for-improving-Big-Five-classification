import warnings
warnings.filterwarnings("ignore")


'''
from pinecone import Pinecone, ServerlessSpec

pinecone = Pinecone(
    api_key="8caa514e-889a-4166-9319-d91f0925fc9d"
)

spec = ServerlessSpec(cloud='aws', region='us-east-1') 

# Crear un índice de Pinecone si no existe
index_name = "chat-history"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(name=index_name, spec=spec, dimension=1536)

# Conectar al índice
index = pinecone.Index(index_name)

from langchain.vectorstores import Pinecone
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx
from langchain.chains import GraphQAChain
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import OpenAI

# conexiòn a modelo local
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
messages = [
    {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}
]

# Texto de ejemplo para la creación del grafo
text = """
Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
She was, in 1906, the first woman to become a professor at the University of Paris. 
"""

def process_response(document):
    nodes_set = set()
    relationships_set = set()

    # Accede al contenido del documento
    print("Metadata del documento:", document.metadata)
    relationships = document.metadata.get("relationships", [])
    for rel in relationships:
        tail = rel.get("tail")
        tail_type = rel.get("tail_type")
        if tail and tail_type:
            nodes_set.add((tail, tail_type))
        else:
            print(f"Missing 'tail' or 'tail_type' in relationship: {rel}")

    return {"nodes": nodes_set, "relationships": relationships_set}

documents = [Document(page_content=text)]
print("Documentos creados\n", documents)

# Transforma los documentos en grafos
llm_transformer = LLMGraphTransformer(llm=client)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Utiliza el método modificado
graph_documents = []
for document in documents:
    response = process_response(document)
    print(response)
    graph_documents.append(response)

print("Documentos transformados en grafos\n", graph_documents)

# # Filtra los nodos y relaciones permitidos
# llm_transformer_filtered = LLMGraphTransformer(
#     llm=client,
#     allowed_nodes=["Person", "Country", "Organization"],
#     allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
# )
# graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)

# Configura el grafo
graph = NetworkxEntityGraph()
print("Grafo configurado\n")

# Agrega nodos al grafo
for graph_doc in graph_documents:
    for node in graph_doc["nodes"]:
        graph.add_node(node[0])

# Agrega relaciones al grafo
for graph_doc in graph_documents:
    for rel in graph_doc["relationships"]:
        graph.add_edge(rel["source"], rel["target"], relation=rel["type"])

print("Nodos y relaciones agregados al grafo\n", graph._graph.nodes, graph._graph.edges)

# for node in graph_documents_filtered[0].nodes:
#     graph.add_node(node.id)

# for edge in graph_documents_filtered[0].relationships:
#     graph._graph.add_edge(
#         edge.source.id,
#         edge.target.id,
#         relation=edge.type,
#     )

# Crea el GraphQAChain
chain = GraphQAChain.from_llm(
    llm=client, 
    graph=graph, 
    verbose=True
)
print("Chain creado\n")

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Pinecone(index=index, embedding=hf_embeddings.embed_query, text_key="text")

# Almacenar los nodos y sus relaciones como contexto en Pinecone
for node in graph._graph.nodes:
    context = {
        "id": node,
        "relationships": list(graph._graph.edges(node, data=True))
    }
    vectorstore.add_documents([Document(page_content=str(context))])

# Función para buscar en la memoria con Pinecone
def search_memory(query):
    results = vectorstore.similarity_search(query, k=5)
    print("Resultados de la búsqueda en la memoria:", results)
    return results

# Ejemplo de uso
query = "Tell me about Marie Curie's husband"
results = search_memory(query)
for result in results:
    print(result.page_content)
'''

import os
import pinecone
from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_community.document_loaders import DirectoryLoader
from pinecone import Pinecone, ServerlessSpec

# Cargar documentos (compartido para todos los usuarios)
documents = DirectoryLoader("/home/giu/PFC2/LLMs-for-improving-Big-Five-classification/HOPE_WSDM_2022/All").load()

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
messages = [
    {"role": "system", "content": '''You are a psychotherapist conducting a conversation with a client. Your main goals are to gather information about the client's personality and avoid dataset bias. 
Use the retrieved examples to help form your questions, ensuring they are reflective, empathetic, and encourage deep introspection.

1. Formulate your responses as insightful questions or statements to guide the client's self-reflection.
2. Avoid repeating questions from the conversation history.
3. Ensure your questions are designed to elicit detailed and unbiased responses, avoiding yes or no questions.
4. Use the Socratic method to explore the client's thoughts, feelings, and motivations.
5. If the client provides limited information, ask open-ended questions to gather more details.
6. Avoid using excessive negative language unless there are clear indicators of neuroticism. Strive to use neutral or positive wording where appropriate to avoid misinterpreting the client's statements. Ensure the use or avoidance of certain words according to the client's detected traits, to make them more evident.
7. Your goal is to collect sufficient information in the areas of openness, conscientiousness, extraversion, agreeableness, and neuroticism.

Only respond with the next question or statement you would make as a psychotherapist. Once you have gathered enough information in all five areas, respond with 'Enough'. Ensure the conversation does not exceed 2048 tokens. The client may respond in English or Spanish.
'''}
]

pinecone = Pinecone(api_key="8caa514e-889a-4166-9319-d91f0925fc9d")
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Configurar el vectorstore de documentos (compartido para todos los usuarios)
docs_index_name = "docs-memory"
if docs_index_name not in pinecone.list_indexes().names():
    pinecone.create_index(name=docs_index_name, spec=spec, dimension=384)
docs_index = pinecone.Index(docs_index_name)

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs_vectorstore = LangchainPinecone(index=pinecone.Index(docs_index_name), embedding=hf_embeddings.embed_query, text_key="text")


# Función para crear/obtener el vectorstore del chat según el usuario
def get_chat_vectorstore_for_user(username):
    chat_index_name = f"{username}-mem"

    # Crear el índice para el chat si no existe
    if chat_index_name not in pinecone.list_indexes().names():
        pinecone.create_index(name=chat_index_name, spec=spec, dimension=384)
    chat_index = pinecone.Index(chat_index_name)

    # Configurar el vectorstore del chat
    chat_vectorstore = LangchainPinecone(index=pinecone.Index(chat_index_name), embedding=hf_embeddings.embed_query, text_key="text")

    return chat_vectorstore


WINDOW_SIZE = 8

def send_message(user_content, messages, chat_vectorstore):
    messages.append({"role": "user", "content": user_content})

    # Recuperar interacciones previas
    previous_messages = load_previous_interactions(user_content, chat_vectorstore)

    # Recuperar ejemplos relevantes del RAG (documentos) que contengan preguntas/comentarios del psicoterapeuta
    relevant_docs = retrieve_relevant_documents(user_content)
    
    # Unir interacciones previas y mensajes recientes
    all_messages = previous_messages + messages[-WINDOW_SIZE:]

    # Adjuntar los documentos relevantes al prompt para proporcionar contexto adicional
    for doc in relevant_docs:
        all_messages.append({"role": "system", "content": f"(Document): {doc}"})

    # Generar la respuesta del modelo
    completion = client.chat.completions.create(
        model="local-model",
        messages=all_messages,
        temperature=0.7,
    )

    # Obtener la respuesta del asistente
    assistant_message = completion.choices[0].message.content

    # Añadir la respuesta del asistente al historial
    messages.append({"role": "assistant", "content": assistant_message})

    # Almacenar la respuesta en el vectorstore del chat
    store_in_chat_vectorstore(user_content, assistant_message, chat_vectorstore)

    return assistant_message

def store_in_chat_vectorstore(user_content, assistant_message, chat_vectorstore):
    # Crea un vector para la consulta
    vector = hf_embeddings.embed_query(assistant_message)

    # Prepara el documento para almacenar en la memoria del chat
    document_id = f"msg-{len(messages)}"  # Genera un ID único
    chat_vectorstore.add_texts([assistant_message], metadatas=[{"user_input": user_content}], ids=[document_id])

def load_previous_interactions(message, chat_vectorstore):
    """Carga interacciones anteriores desde el vectorstore del chat."""
    results = chat_vectorstore.similarity_search(message)  # Busca mensajes almacenados
    previous_messages = []
    for result in results:
        previous_messages.append({"role": "user", "content": result.metadata['user_input']})
        previous_messages.append({"role": "assistant", "content": result.page_content})
    return previous_messages

def retrieve_relevant_documents(query):
    """Busca ejemplos relevantes de los documentos cargados para el RAG que contengan preguntas o comentarios del psicoterapeuta."""
    results = docs_vectorstore.similarity_search(query)
    
    # Filtrar resultados que contengan preguntas/comentarios del psicoterapeuta
    relevant_docs = []
    for result in results:
        if "psychotherapist:" in result.page_content:  # Ajusta según cómo esté estructurado el contenido
            relevant_docs.append(result.page_content)
    
    return relevant_docs[:3]  # Devolver los tres documentos más relevantes

# Almacenar los documentos en el vectorstore compartido una sola vez
def store_documents_in_vectorstore_once():
    if docs_vectorstore._index.describe_index_stats()["total_vector_count"] == 0:
        print("Guardando documentos en el vectorstore...")
        texts = [doc.page_content for doc in documents]
        docs_vectorstore.add_texts(texts, metadatas=[{"source": "document"} for _ in texts])
    else:
        print("Documentos ya almacenados.")

# Almacenar los documentos una sola vez al inicio
store_documents_in_vectorstore_once()

# Iniciar la conversación preguntando por el usuario
username = input("Introduce tu nombre de usuario: ")

# Obtener el vectorstore del chat específico para el usuario
chat_vectorstore = get_chat_vectorstore_for_user(username)

initial_prompt = "Hola, cómo te sientes hoy?"
print(initial_prompt)
messages.append({"role": "assistant", "content":  "Hola, cómo te sientes hoy?"})

# Ejemplo de uso
conversation = []
answers = []
counter = 0
conversation.append(initial_prompt)
while True:
    user_input = input("Tú: ")
    counter += 1
    conversation.append(f"Client: {user_input}")
    answers.append(user_input)
    if user_input.lower() in ["salir", "exit"]:
        print("Saliendo...")
        break
    if counter == 6:
        print("Has alcanzado el límite de respuestas.")
        break
    else:
        response = send_message(user_input, messages, chat_vectorstore)
        print("Asistente:", response)
        conversation.append(f"Psychotherapist: {response}")


import csv

# Save conversation to TXT
conversation_text = "\n\n" + "\n".join(conversation)
with open("conversation.txt", "a") as file:
    file.write(conversation_text + "\n\n")
# Save answers to CSV
answer_text = " ".join(answers)
csv_file = "conversation"+username+".csv"
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([answer_text])  # Write answers
