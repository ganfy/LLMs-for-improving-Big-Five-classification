import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import numpy as np

if len(sys.argv) != 3:
    print("Uso: python script.py <path_dataset> <output_model_file>")
    sys.exit(1)

data_path = sys.argv[1]
output_model_file = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Cargar los datos
data = pd.read_csv(data_path, encoding='latin1')

# Convertir etiquetas 'y' y 'n' a 1.0 y 0.0
data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']] = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].applymap(lambda x: 1.0 if x == 'y' else 0.0)

class PersonalityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.TEXT
        self.labels = dataframe[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        labels = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 2e-5

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

train_dataset = PersonalityDataset(train_data, tokenizer, MAX_LEN)
val_dataset = PersonalityDataset(val_data, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Labels: ", train_dataset.labels)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, data_loader, optimizer, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)

# Función de evaluación
def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            losses.append(loss.item())

    return np.mean(losses)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f'Train loss {train_loss}')

    val_loss = eval_model(model, val_loader, device)
    print(f'Val loss {val_loss}')
    print()

model.save_pretrained(output_model_file)
tokenizer.save_pretrained(output_model_file)

print(f"Modelo guardado correctamente en {output_model_file}.")
