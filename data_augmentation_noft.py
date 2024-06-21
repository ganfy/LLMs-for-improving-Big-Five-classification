import pandas as pd
from langchain.llms import Ollama
import torch
from tqdm import tqdm

# Verificación y configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Configuración del modelo Llama3 para usar la GPU si está disponible
class Llama3Augmenter:
    def __init__(self, model_name):
        self.model = Ollama(model=model_name)

    def augment(self, text):
        try:
            prompt = f"Reescribe este texto manteniendo el tono y mensaje: {text}"
            response = self.model.invoke(prompt)
            augmented_text = response
            return augmented_text
        except Exception as e:
            print(f"Error durante la generación de texto aumentado: {e}")
            return text

# Cargar datos del CSV
# data = pd.read_csv('/content/drive/MyDrive/PFC II/essays.csv', encoding='latin1')
data = pd.read_csv('/home/giu/PFC2/LLMs-for-improving-Big-Five-classification/essays/essays.csv', encoding='latin1')

# Transformación de columnas específicas
data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']] = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].applymap(lambda x: 1.0 if x == 'y' else 0.0)

# Instanciar el aumentador
augmenter = Llama3Augmenter(model_name="phi3")

# Contar las clases iniciales
class_counts = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].sum().to_dict()
umbral_minimo = 20

# Realizar data augmentation para equilibrar las clases
for index, row in tqdm(data.iterrows(), total=len(data), desc="Procesando filas"):
    original_text = row['TEXT']
    label_values = row[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].values

    if any(label_values < umbral_minimo):
        augmented_text = augmenter.augment(original_text)
        new_row = {'TEXT': augmented_text, 'cEXT': label_values[0], 'cNEU': label_values[1],
                   'cAGR': label_values[2], 'cCON': label_values[3], 'cOPN': label_values[4]}
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

# Contar las clases después del data augmentation
final_class_counts = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].sum().to_dict()

# Imprimir los resultados
print("Clases originales:")
print(class_counts)

print("\nClases después del data augmentation:")
print(final_class_counts)

# Guardar el dataframe resultante en un nuevo archivo CSV
output_file = '/content/drive/MyDrive/PFC II/essays_augmented.csv'
data.to_csv(output_file, index=False, encoding='latin1')

print(f"\nDataset aumentado guardado en {output_file}")
