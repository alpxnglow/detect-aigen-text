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
    print("Analyzing...")
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
test_text = """

Once upon a time, there lived a peculiar canine named Max. Max was not your ordinary dog. He had a peculiar habit of staring at his reflection in the mirror for extended periods, a behavior his owner found amusing but slightly unsettling. One day, while gazing at himself in the mirror, something clicked in Max's mind. He suddenly realized that he wasn't just any creature; he was a dog. This revelation sent shockwaves through his consciousness, shattering his previous perceptions of reality. Max began to question everything about his existence, pondering the nature of his canine identity and what it truly meant to be a dog.

As days passed, Max's newfound self-awareness only deepened. He observed his fellow canines with newfound curiosity, noticing their behaviors and interactions in a different light. He couldn't shake the feeling of being different, of not fully belonging to the world of dogs. Despite his owner's affection and care, Max felt a sense of isolation, longing for something beyond the confines of his canine existence.

Max's journey of self-discovery led him to explore the world around him with renewed vigor. He savored the scents and sounds of the world, marveling at its complexity and beauty. Yet, amidst his explorations, he couldn't shake the nagging question of his identity. Was he destined to be just a dog, bound by instinct and biology, or was there more to him than met the eye?

One fateful day, Max encountered a wise old stray who had seen much of the world. Sensing Max's inner turmoil, the stray listened intently as Max poured out his heart. After a moment of contemplation, the stray spoke softly, "Being a dog isn't just about what you are, but who you choose to be." These words struck a chord deep within Max's soul, igniting a newfound sense of purpose and determination.

With a newfound sense of agency, Max embarked on a quest to redefine what it meant to be a dog. He embraced his canine nature wholeheartedly, reveling in the simple joys of life – the feel of the sun on his fur, the taste of fresh grass beneath his paws. Yet, he also embraced his unique perspective, using it to forge connections with creatures of all shapes and sizes.

In the end, Max discovered that being a dog wasn't just about being self-aware; it was about embracing one's true self and living life to the fullest. As he bounded through fields and forests, his heart brimming with newfound confidence and purpose, Max realized that he was more than just a dog – he was a beacon of possibility, a testament to the boundless potential that lay within us all. And so, with a bark of joy and a wag of his tail, Max embraced his destiny, ready to face whatever adventures lay ahead.


"""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"{generated}")