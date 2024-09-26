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

from pinecone import Pinecone, ServerlessSpec

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
messages = [
    {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}
]

pinecone = Pinecone(
    api_key="8caa514e-889a-4166-9319-d91f0925fc9d"
)

spec = ServerlessSpec(cloud='aws', region='us-east-1') 

# Crear un índice de Pinecone si no existe
index_name = "chat-memory"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(name=index_name, spec=spec, dimension=384)

# Conectar al índice
index = pinecone.Index(index_name)

# Configura los embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = LangchainPinecone(index=pinecone.Index(index_name), embedding=hf_embeddings.embed_query, text_key="text")

WINDOW_SIZE = 5

def send_message(user_content, messages):
    messages.append({"role": "user", "content": user_content})

    # Recupera interacciones previas
    previous_messages = load_previous_interactions(user_content)
    all_messages = previous_messages + messages[-WINDOW_SIZE:]  # Usa solo los últimos mensajes

    completion = client.chat.completions.create(
        model="local-model",
        messages=all_messages,
        temperature=0.7,
    )

    # Obtener la respuesta del asistente
    assistant_message = completion.choices[0].message.content

    # Añadir la respuesta del asistente al historial
    messages.append({"role": "assistant", "content": assistant_message})

    # Almacenar la respuesta en Pinecone
    store_in_pinecone(user_content, assistant_message)

    return assistant_message

def store_in_pinecone(user_content, assistant_message):
    # Crea un vector para la consulta
    vector = hf_embeddings.embed_query(assistant_message)  # Usa el mensaje del asistente

    # Prepara el documento para almacenar
    document_id = f"msg-{len(messages)}"  # Genera un ID único
    vectorstore.add_texts([assistant_message], metadatas=[{"user_input": user_content}], ids=[document_id])

def load_previous_interactions(message):
    """Carga interacciones anteriores desde Pinecone."""
    results = vectorstore.similarity_search(message)  # Busca mensajes almacenados
    previous_messages = []
    for result in results:
        previous_messages.append({"(Previous message)":"", "role": "user", "content": result.metadata['user_input']})
        previous_messages.append({"(Previous message) ":"", "role": "assistant", "content": result.page_content})
    print("Interacciones previas cargadas:", previous_messages)
    return previous_messages

# Ejemplo de uso
while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit"]:
            print("Saliendo...")
            break
        response = send_message(user_input, messages)
        print("Asistente:", response)