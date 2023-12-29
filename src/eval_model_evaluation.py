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

# Evaluate the performance of the model
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
            except pd.errors.ValueError as e:
                print("evaluate: value error encountered while evaluation")
                print(e)
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

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

accuracy, report = evaluate(model, val_dataloader, device)
print(f"Validation Accuracy: {accuracy:.4f}")
print(report)
