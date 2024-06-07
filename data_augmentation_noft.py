from llama_index.core import Settings, PromptTemplate
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import pandas as pd


llm = Ollama(model="llama3", request_timeout=120.0)
Settings.llm = llm
Settings.chunk_size = 512

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents([], storage_context=storage_context)

class Llama3Augmenter:
    def __init__(self, llm_model, query_engine):
        self.llm_model = llm_model
        self.query_engine = query_engine

    def augment(self, text):
        try:
            system_prompt = (
                "Reescribe este texto: "
                "{query_str}"
            )
            query_wrapper_prompt = PromptTemplate(system_prompt)
            query = query_wrapper_prompt.format(query_str=text)
            response = self.query_engine.query(query)
            augmented_text = response['results'][0]['text']
            return augmented_text
        except Exception as e:
            print(f"Error durante la generación de texto aumentado: {e}")
            return text


data = pd.read_csv('essays/essays.csv', encoding='latin1')
data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']] = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].applymap(lambda x: 1.0 if x == 'y' else 0.0)

query_engine = index.as_query_engine(similarity_top_k=10)
llama3_augmenter = Llama3Augmenter(llm, query_engine)

class_counts = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].sum().to_dict()

umbral_minimo = 20

for index, row in data.iterrows():
    original_text = row['TEXT']
    label_values = row[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].values

    if sum(label_values) < umbral_minimo:
        augmented_text = llama3_augmenter.augment(original_text)
        new_row = {'TEXT': augmented_text, 'cEXT': label_values[0], 'cNEU': label_values[1],
                   'cAGR': label_values[2], 'cCON': label_values[3], 'cOPN': label_values[4]}
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

final_class_counts = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].sum().to_dict()

print("Clases originales:")
print(class_counts)

print("\nClases después del data augmentation con llama3:")
print(final_class_counts)

output_file = 'essays_augmented.csv'
data.to_csv(output_file, index=False)

print(f"\nDataset aumentado guardado en {output_file}")

