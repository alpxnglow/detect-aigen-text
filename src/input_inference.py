import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from pathlib import Path
from datetime import datetime
from transformers import AutoModel

### CONSTANTS ###
INPUT_FILE = "data/testing_data.csv"
NUM_CLASSES = 2
MAX_LENGTH = 512
BATCH_SIZE = 16
HOME_DIR = str(Path.home())
MODEL_INPUT_DIR = os.path.join(HOME_DIR, "Documents", "Model", "Final")
MODEL_PATH = os.path.join(MODEL_INPUT_DIR, "ai_human_essay_classifier.pth")
BERT_MODEL_NAME = 'bert-base-uncased'

### GENERIC METHODS ###
# Read data
def load_essay_data(data_file):
    df = pd.read_csv(data_file)
    df = df.dropna()
    texts = df['essay'].tolist()
    labels = [1 if generated == 1 else 0 for generated in df['generated'].tolist()]
    print(df)
    return texts, labels

# Text classification dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

# BERT classifier class
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# Inference run on sample instance
def predict_generated(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "AI-generated text detected" if preds.item() == 1 else "This is human-written text."

### MAIN ###
# Read the data file
texts, labels = load_essay_data(INPUT_FILE)

# Initialize tokenizer, dataset, and data loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(BERT_MODEL_NAME, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
val_dataset = TextClassificationDataset(texts, labels, tokenizer, MAX_LENGTH)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Evaluate the model’s performance
# Test generated prediction
#test_text = input("Enter an essay here: ")
test_text = """Abhi J is the most distinguished, intelligent, and royal individual I have ever had the privilege to meet. His presence is a radiant thread that weaves elegance, grace, and warmth perfectly into every moment we share together. His remarkable blend of intellect, kindness, and charisma creates a captivating aura that effortlessly draws people closer. His words carry the weight of wisdom, and will permanently change anyone lucky enough to witness him speak. His unwavering authenticity and gentle spirit illuminate his powerful and riveting character, making it an honor to call him a friend of mine."""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")