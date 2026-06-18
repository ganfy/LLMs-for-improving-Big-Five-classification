import os
import pinecone
from openai import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain_community.document_loaders import DirectoryLoader
from pinecone import Pinecone, ServerlessSpec
from langchain.memory import ConversationKGMemory
import warnings
warnings.filterwarnings("ignore")


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


WINDOW_SIZE = 8


from langchain.llms.base import BaseLanguageModel
from langchain.schema import BaseMessage
import json
from pydantic import BaseModel
from typing import Optional


class LocalLLM(BaseLanguageModel, BaseModel):
    client: Optional[object] = None  # Hacer 'client' opcional y darle un valor por defecto

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'client' in kwargs:
            self.client = kwargs['client']
        else:
            self.client = None  # O puedes establecerlo a un valor predeterminado

    def _call(self, prompt):
        response = self.client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content

    def agenerate(self, prompts, **kwargs):
        """Generate responses for multiple prompts."""
        results = []
        for prompt in prompts:
            result = self._call(prompt)
            results.append(result)
        return results

    def generate(self, prompt, **kwargs):
        """Generate a single response."""
        return self._call(prompt)

    def predict(self, prompt, **kwargs):
        """Return the same as generate."""
        return self.generate(prompt, **kwargs)

    def predict_messages(self, messages, **kwargs):
        """Convert messages to prompts and generate responses."""
        prompt = self._messages_to_prompt(messages)
        return self._call(prompt)

    def _messages_to_prompt(self, messages):
        """Convert a list of messages to a single string prompt."""
        return '\n'.join([f"{msg['role']}: {msg['content']}" for msg in messages])

    @property
    def llm_type(self):
        return "local_llm"
 
    def agenerate_prompt(self, messages, **kwargs):
        """Convierte los mensajes a un prompt que el modelo pueda procesar."""
        return self._messages_to_prompt(messages)

    def apredict(self, prompt, **kwargs):
        """Versión asincrónica de `predict`."""
        return self.generate(prompt, **kwargs)

    def apredict_messages(self, messages, **kwargs):
        """Versión asincrónica de `predict_messages`."""
        prompt = self.agenerate_prompt(messages, **kwargs)
        return self._call(prompt)

    def generate_prompt(self, inputs: dict, callbacks: Optional[list] = None) -> str:
            if callbacks:
                for callback in callbacks:
                    pass
            
            prompt = inputs.get("input", "")
            return f"Generate a response for: {prompt}"

    def invoke(self, prompt, **kwargs):
        """Este es el método que hace la llamada al modelo para generar la respuesta."""
        return self._call(prompt)


from langchain_huggingface import HuggingFacePipeline

from transformers import pipeline

# Crear el pipeline de transformers con un límite mayor de tokens
text2text_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

# Configurar el límite de tokens a un valor mayor
# text2text_pipeline.model.config.max_length = 2000

# Adaptar el pipeline a LangChain
from langchain.llms import HuggingFacePipeline

local_llm = HuggingFacePipeline(pipeline=text2text_pipeline)



    
# local_llm = LocalLLM(client=client) # didn't work



def send_message(user_content, messages, chat_vectorstore):
    messages.append({"role": "user", "content": user_content})

    # Recuperar interacciones previas desde el grafo
    previous_messages = chat_vectorstore.load_memory_variables({"input": user_content})['history']

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

    # Almacenar la interacción en la memoria de grafo
    chat_vectorstore.save_context({"input": user_content}, {"output": assistant_message})

    return assistant_message


def store_in_chat_vectorstore(user_content, assistant_message, chat_vectorstore):
    
    # Crea un vector para la consulta
    vector = hf_embeddings.embed_query(assistant_message)

    # Prepara el documento para almacenar en la memoria del chat
    document_id = f"msg-{len(messages)}"  # Genera un ID único
    chat_vectorstore.add_texts([assistant_message], metadatas=[{"user_input": user_content}], ids=[document_id])

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
chat_vectorstore = ConversationKGMemory(llm= local_llm, return_messages=True)

initial_prompt = "Hola, cómo te sientes hoy?"
print(initial_prompt)
messages.append({"role": "assistant", "content":  "Hola, cómo te sientes hoy?"})

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # Código que genera la advertencia
            response = send_message(user_input, messages, chat_vectorstore)

        print("Asistente:", response)
        conversation.append(f"Psychotherapist: {response}")


import csv

# Save conversation to TXT
conversation_text = "\n\n" + "\n".join(conversation)
with open("conversationGraph.txt", "a") as file:
    file.write(conversation_text + "\n\n")
# Save answers to CSV
answer_text = " ".join(answers)
csv_file = "conversationGraph"+username+".csv"
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([answer_text])  # Write answers

