# from openai import OpenAI
# import firebase_admin
# from firebase_admin import credentials,db


# cred = credentials.Certificate("google_service.json")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://empre-f29ac-default-rtdb.firebaseio.com/'
# })

# # Acceder a un nodo específico en el árbol de datos
# ref = db.reference('mensaje')
# awns = {'msg':"hol",}
# ref.update(awns)

# ref = db.reference('respuesta')
# awns = {'anwser':"",}
# ref.update(awns)
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
# messages = [
#     {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}
# ]
# count =0
# def send_message(user_content):
#     # Añadir el mensaje del usuario al historial
#     global messages
#     global count  
#     messages.append({"role": "user", "content": user_content})

#     # Crear la respuesta basada en el historial de mensajes
#     completion = client.chat.completions.create(
#         model="local-model",
#         messages=messages,  # Pasa el historial completo aquí
#         temperature=0.7,
#     )

#     # Obtener la respuesta del asistente
#     assistant_message = completion.choices[0].message.content
#     count =count+1
#     # Añadir la respuesta del asistente al historial
#     messages.append({"role": "assistant", "content": assistant_message})
#     if(count==5):
#        count=0
#        messages = [
#        {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}]
#     # Devolver la respuesta del asistente
#     return assistant_message

# # Ejemplo de uso
# temp="hol"
# while(True):
#   ref = db.reference('mensaje')
#   datos = ref.get()
#   mensaje = datos.get('msg')
#   if(mensaje == "exit"):
#     break
#   if(mensaje == "rest"):
#     ref = db.reference('mensaje')
#     awns = {'msg':"hol",}
#     ref.update(awns)
#     ref = db.reference('respuesta')
#     awns = {'anwser':"",}
#     ref.update(awns)
#     temp ="hol"
#     count=0
#   if(mensaje != temp):
#     print(mensaje)
#     respuesta = send_message(mensaje)
#     ref = db.reference('respuesta')
#     awns = {'anwser':respuesta,}
#     ref.update(awns)
#     #print(respuesta)
#     temp = mensaje


from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
from qdrant_client.http import models
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client.models import VectorParams, Distance
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Conectar con OpenAI
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
messages = [
    {"role": "system", "content": "Eres un asistente que ofrece información certera y no muy extensa sobre los temas consultados."}
]

# Cargar el modelo de embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Configurar Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=60.0)  # 60 segundos de timeout
collection_name = "chatbot_memory"

if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Crear un objeto de embeddings usando LangChain y Hugging Face
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
vector_store = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=hf_embeddings)

def get_relevant_memory(content):
    # Buscar mensajes relevantes en la memoria
    docs = vector_store.similarity_search(content, k=5)
    return docs

def update_reference_count(doc_id):
    # Incrementar la cuenta de referencias
    payload = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter={"must": [{"key": "_id", "match": {"value": doc_id}}]}
    )
    if payload:
        updated_references = payload[0]["references"] + 1
        qdrant_client.update_payload(
            collection_name=collection_name,
            payload={"references": updated_references},
            points_selector=models.PointIdsList(ids=[doc_id])
        )

def forget_unused_memory(threshold=1):
    # Eliminar mensajes poco referenciados
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="references",
                        match=models.RangeCondition(lt=threshold)
                    )
                ]
            )
        )
    )


import uuid

def store_message_in_memory(text):
    point_id = str(uuid.uuid4())
    vector = hf_embeddings.embed_query(text)
    # vector_store.add_texts(texts=[text], ids=[point_id])
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[models.PointStruct(id=point_id, vector=vector, payload={"text": text})]
    )

def retrieve_relevant_messages(query, top_k=3):
    query_vector = hf_embeddings.embed_query(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    # Verifica que cada punto tenga el campo 'text' en su payload antes de añadirlo a los resultados
    return [hit.payload.get("text", "") for hit in search_result if "text" in hit.payload]


def send_message(user_content):
    global messages
    messages.append({"role": "user", "content": user_content})

    # Guardar el mensaje en la memoria
    store_message_in_memory(user_content)

    # Recuperar mensajes relevantes de la memoria
    relevant_messages = retrieve_relevant_messages(user_content)

    # Añadir mensajes relevantes al historial como contexto
    for msg in relevant_messages:
        if msg:  # Solo añadir si hay contenido en el mensaje
            messages.append({"role": "system", "content": f"[Contexto]: {msg}"})

    # Verificar el formato de mensajes antes de enviarlo
    formatted_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages if isinstance(msg, dict)]

    # Imprimir el contenido de los mensajes para depurar
    print("Mensajes enviados a la API:", formatted_messages)

    # Generar la respuesta basada en el historial
    completion = client.chat.completions.create(
        model="local-model",
        messages=formatted_messages,  # Pasa el historial completo aquí
        temperature=0.7,
    )

    assistant_message = completion.choices[0].message.content

    # Añadir la respuesta del asistente al historial
    messages.append({"role": "assistant", "content": assistant_message})

    # Guardar la respuesta en la memoria
    store_message_in_memory(assistant_message)

    return assistant_message


def main():
    print("¡Bienvenido al chatbot! Escribe 'salir' para terminar la conversación.")
    
    while True:
        user_message = input("Tú: ")
        
        if user_message.lower() == "salir":
            print("¡Adiós!")
            break
        
        response = send_message(user_message)
        print(f"Asistente: {response}")

if __name__ == "__main__":
    main()