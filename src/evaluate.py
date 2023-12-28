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

# Inference run on sample instance
def predict_generated(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "1" if preds.item() == 1 else "0"

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

# Evaluate the model’s performance
# Test generated prediction
test_text = """Income inequality can affect our health and how long we live. It's important to understand how money is distributed in our society and how it affects healthcare disparities and life expectancy. Scientists have done studies and found that there is a connection between income inequality and health.

First, let's talk about healthcare disparities. When there is a big gap between how much money people earn, it can lead to differences in access to healthcare. People who have less money might struggle to afford healthcare services like doctor visits, medicine, and surgeries. On the other hand, people with more money can afford better healthcare and have more options available to them.

In a study conducted by researchers, they found that people with lower incomes are more likely to have chronic diseases such as diabetes and heart disease. This could be due to the lack of access to healthcare and preventive services. Without proper healthcare, these chronic diseases can become more severe and lead to a shorter life expectancy.

Life expectancy is another area where income inequality plays a role. Life expectancy refers to how long a person is expected to live on average. Studies have shown that in countries with higher income inequality, people tend to have a lower life expectancy compared to countries with less income inequality.

There are a few reasons why income inequality can impact life expectancy. One reason is that people with lower incomes might not have access to healthy food options and live in neighborhoods with poor environmental conditions. These factors can lead to unhealthy lifestyles and increase the risk of developing diseases.

Another reason is the stress that comes with living in poverty. People who are struggling financially might constantly worry about money, and this stress can have a negative impact on their health. It can lead to mental health issues and physical health problems, ultimately reducing life expectancy.

Socioeconomic factors also contribute to the connection between income inequality and health. Socioeconomic factors include things like education level, occupation, and social status. People with higher socioeconomic status tend to have better health outcomes and longer life expectancy.

In conclusion, income inequality has a significant impact on healthcare disparities and life expectancy. People with lower incomes often face barriers in accessing quality healthcare, which can lead to poorer health outcomes and shorter life expectancy. Socioeconomic factors also come into play, as they contribute to the disparities in health outcomes. It is important to address income inequality and work towards creating a more equitable society where everyone has equal access to healthcare and opportunities for a healthy life."""
generated = predict_generated(test_text, model, tokenizer, device)
print(f"1. Predicted generated: {generated}")

# Test generated prediction
test_text = """
The experiment aimed to investigate conjugation and recombination in Esherichia coli and determine the chromosomal order of a number of genes for amino acid synthesis and sugar metabolism. This was carried out by mixing donor and recipient strains and at certain time intervals interrupting the mating and plating on selective media. Qualitative assessment of the plates was then used to map the gene order. The chromosomal gene order was found to be thr, arg and xyl, ilv, leu, pro, his (see discussion for explanation of gene names).

Introduction: Bacterial DNA can be exchanged between different strains allowing new allele combinations to be produced. This means that a 'wild type' gene can be passed to a strain with a mutation in this gene and through recombination the gene activity can be restored. The most common mechanisms of genetic exchange are transformation, transduction and conjugation. In each case genetic material is transferred from the donor strain to the recipient strain and this is assessed by monitoring a change in phenotype of the recipient. This is usually carried out by arranging growth conditions to select for recipients that have received the genetic material as in bacteria the change is normally in a requirement for an amino acid, nucleotide or vitamin, and ability to use a compound as an energy source or a resistance to an antibacterial agent. Here conjugation will be looked at. This involves cell-cell contact between donor and recipient achieved through an F pilus (a protein appendage) synthesized by the donor and used to anchor the recipient to form a mating pair (or mating aggregate). Here conjugation will be investigated in Escherichia coli. In E. coli conjugation involves the F factor, which is present in the donor but not recipient strain (F-). This experiment involves conjugation between an Hfr (high frequency recombination - the F factor is integrated into the chromosome) strain, the donor, and a F- strain, the recipient. The E. coli chromosome is circular and transfer always starts from the same point, the F factor integration site, which allows time of entry mapping to be carried out: As soon as the two strains are mixed mating aggregates will form and transfer of the chromosome will start from a fixed site and in a fixed orientation. If the mating pairs are physically interrupted at intervals, mapping the time of entry of genes from the donor into the recipient can create a genetic map of the chromosome. The aim of the experiment was to understand these concepts and methods of bacterial genetics by exchanging pieces of E. coli chromosome between different strains by the process of conjugation and using a non-quantitative method to establish the order of some genes relating to amino acid metabolism and sugar catabolism.

Method: The experiment was carried out as laid out in the lab manual with the following detail. The donor strain used was E. coli KL14 (thi-1 Hfr KL14) The bacteria donor and recipient strains were allowed to grow in the shaking water bath at 37 oC for 120 minutes before being mixed together.

Results: From the results it is clear that the bacteria could grow earlier on some plates than others. In this case the sample selective media lacking threonine (Plate 6) grew first at time 0minutes (but also had growth on the recipient strain area), then on both plates lacking arginine (plate 1) and with xylose as the sugar (plate 7) at 15 minutes. The sample on plate 3, lacking isoleucine and valine was next to grow then plate 4 lacking leucine at 30 minutes. Plate 5 lacking proline showed growth at 60 minutes and finally plate 2 lacking histidine at 120minutes. The E. coli chromosomal gene order was determined to be: thr, arg and xyl, ilv, leu, pro, his.

Discussion: From the table of results it is clear that the genes allowing growth to occur by transferring the wild type allele were transferred at different time points for each gene selected. This is due to the process of Hfr conjugation discussed previously in which the chromosome is passed into the recipient in a certain order. As the mating pairs were disrupted at the time intervals given above no further gene transfer could take place in that particular sample and the sample would only grow on a selective plate if the gene allowing synthesis of that particular amino acid or metabolism of the sugar had already been passed into the recipient strain. It was therefore clear that by looking at the time points at which the samples started growing on different plates it was possible to map the order in which the genes had been transferred from the Hfr strain and hence the order in which they are present on the chromosome. Both plate 1 and 7 (lacking arginine and with xylose) showed growth at 15mins and had roughly the same amount of growth at this time and following times. It can therefore be concluded that these genes transferred at times fairly close together between 0 and 15 minutes and are therefore likely to be fairly close together on the chromosome. Plates 3 and 4 (lacking isoleucine/valine and leucine respectively) both showed growth for the first time on the sample plated at 30minutes. However Plate 3 showed more growth at the next sample at 60 minutes (+++) compared to plate 4 (++) and therefore it was concluded that the isoleucine/valine gene would have been the one transferred first probably nearer to the start of the time period (ie. 15minutes) Plate 6, lacking threonine appears to have growth at time 0 and much growth (+ +) at 15 minutes. However this plate also shows growth of the recipient strain alone. This could be due to the mutation in the recipient reverting to the wild type. This is only possible in a point-mutation - a deletion cannot revert as the DNA is missing completely. The mutation in the threonine coding gene in the recipient strain is known to not be a deletion (see materials list in lab manual) and therefore this reversion in possible. This would then invalidate the results for this plate, as this growth does not represent gene transfer. However the growth on plate 6 does still appear to show a gradation increasing at 15 minutes and 30 minutes and therefore there is still a high chance that this is an early gene. Taking all this into account an approximate order of gene transfer from the Hfr strain to the recipient can therefore be determined as follows: thr, arg and xyl, ilv, leu, pro, his. Where thr = gene coding for threonine, arg = gene coding for arginine, xyl = gene involved in xylose metabolism, ilv = gene coding for isoleucine and valine, leu = gene coding for leucine, pro = gene coding for proline, his = gene coding for histidine. Possible sources of errors in this experiment include the risk of dislodging the mating pairs by causing shaking or jarring whilst removing samples. This would have led to no further gene transfer in the bacteria concerned and hence errors to the results as growth may not have occurred at a time point in which it would have otherwise. The time at which the genes appeared to be transferred was determined by plating onto selective media. It was necessary to not only select for the recombinant (the recipient after it has received genetic material) but also against the donor. This is known as counter selection and here was carried out by including nalidixic acid in the medium as this prevents donor growth but not the recipient. Nalidixic acid will also prevent further gene transfer. If it was not included the donor strain would also grow on the selective media and all samples would show growth at all times due to the presence of the donor. The procedure and accuracy of results could be improved by using a quantitative approach in which the number of recombinant colonies are counted at each time point using a viable count. In conclusion the rough mapping of gene order along the E. coli chromosome can be determined qualitatively by physically interrupting mating pairs of a donor and recipient E. coli strain at certain intervals and noting growth levels after plating on selective media. This highlights the basic concepts and methods of conjugation and recombination in bacteria. 
"""
generated = predict_generated(test_text, model, tokenizer, device)
print(f"2. Predicted generated: {generated}")

# # Test generated prediction
# test_text = "Worst movie of the year."
# generated = predict_generated(test_text, model, tokenizer, device)
# print("Worst movie of the year.")
# print(f"Predicted generated: {generated}")