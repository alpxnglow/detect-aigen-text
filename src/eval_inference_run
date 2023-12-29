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
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

# Test generated prediction
test_text = """
The experiment aimed to investigate conjugation and recombination in Esherichia coli and determine the chromosomal order of a number of genes for amino acid synthesis and sugar metabolism. This was carried out by mixing donor and recipient strains and at certain time intervals interrupting the mating and plating on selective media. Qualitative assessment of the plates was then used to map the gene order. The chromosomal gene order was found to be thr, arg and xyl, ilv, leu, pro, his (see discussion for explanation of gene names).

Introduction: Bacterial DNA can be exchanged between different strains allowing new allele combinations to be produced. This means that a 'wild type' gene can be passed to a strain with a mutation in this gene and through recombination the gene activity can be restored. The most common mechanisms of genetic exchange are transformation, transduction and conjugation. In each case genetic material is transferred from the donor strain to the recipient strain and this is assessed by monitoring a change in phenotype of the recipient. This is usually carried out by arranging growth conditions to select for recipients that have received the genetic material as in bacteria the change is normally in a requirement for an amino acid, nucleotide or vitamin, and ability to use a compound as an energy source or a resistance to an antibacterial agent. Here conjugation will be looked at. This involves cell-cell contact between donor and recipient achieved through an F pilus (a protein appendage) synthesized by the donor and used to anchor the recipient to form a mating pair (or mating aggregate). Here conjugation will be investigated in Escherichia coli. In E. coli conjugation involves the F factor, which is present in the donor but not recipient strain (F-). This experiment involves conjugation between an Hfr (high frequency recombination - the F factor is integrated into the chromosome) strain, the donor, and a F- strain, the recipient. The E. coli chromosome is circular and transfer always starts from the same point, the F factor integration site, which allows time of entry mapping to be carried out: As soon as the two strains are mixed mating aggregates will form and transfer of the chromosome will start from a fixed site and in a fixed orientation. If the mating pairs are physically interrupted at intervals, mapping the time of entry of genes from the donor into the recipient can create a genetic map of the chromosome. The aim of the experiment was to understand these concepts and methods of bacterial genetics by exchanging pieces of E. coli chromosome between different strains by the process of conjugation and using a non-quantitative method to establish the order of some genes relating to amino acid metabolism and sugar catabolism.

Method: The experiment was carried out as laid out in the lab manual with the following detail. The donor strain used was E. coli KL14 (thi-1 Hfr KL14) The bacteria donor and recipient strains were allowed to grow in the shaking water bath at 37 oC for 120 minutes before being mixed together.

Results: From the results it is clear that the bacteria could grow earlier on some plates than others. In this case the sample selective media lacking threonine (Plate 6) grew first at time 0minutes (but also had growth on the recipient strain area), then on both plates lacking arginine (plate 1) and with xylose as the sugar (plate 7) at 15 minutes. The sample on plate 3, lacking isoleucine and valine was next to grow then plate 4 lacking leucine at 30 minutes. Plate 5 lacking proline showed growth at 60 minutes and finally plate 2 lacking histidine at 120minutes. The E. coli chromosomal gene order was determined to be: thr, arg and xyl, ilv, leu, pro, his.

Discussion: From the table of results it is clear that the genes allowing growth to occur by transferring the wild type allele were transferred at different time points for each gene selected. This is due to the process of Hfr conjugation discussed previously in which the chromosome is passed into the recipient in a certain order. As the mating pairs were disrupted at the time intervals given above no further gene transfer could take place in that particular sample and the sample would only grow on a selective plate if the gene allowing synthesis of that particular amino acid or metabolism of the sugar had already been passed into the recipient strain. It was therefore clear that by looking at the time points at which the samples started growing on different plates it was possible to map the order in which the genes had been transferred from the Hfr strain and hence the order in which they are present on the chromosome. Both plate 1 and 7 (lacking arginine and with xylose) showed growth at 15mins and had roughly the same amount of growth at this time and following times. It can therefore be concluded that these genes transferred at times fairly close together between 0 and 15 minutes and are therefore likely to be fairly close together on the chromosome. Plates 3 and 4 (lacking isoleucine/valine and leucine respectively) both showed growth for the first time on the sample plated at 30minutes. However Plate 3 showed more growth at the next sample at 60 minutes (+++) compared to plate 4 (++) and therefore it was concluded that the isoleucine/valine gene would have been the one transferred first probably nearer to the start of the time period (ie. 15minutes) Plate 6, lacking threonine appears to have growth at time 0 and much growth (+ +) at 15 minutes. However this plate also shows growth of the recipient strain alone. This could be due to the mutation in the recipient reverting to the wild type. This is only possible in a point-mutation - a deletion cannot revert as the DNA is missing completely. The mutation in the threonine coding gene in the recipient strain is known to not be a deletion (see materials list in lab manual) and therefore this reversion in possible. This would then invalidate the results for this plate, as this growth does not represent gene transfer. However the growth on plate 6 does still appear to show a gradation increasing at 15 minutes and 30 minutes and therefore there is still a high chance that this is an early gene. Taking all this into account an approximate order of gene transfer from the Hfr strain to the recipient can therefore be determined as follows: thr, arg and xyl, ilv, leu, pro, his. Where thr = gene coding for threonine, arg = gene coding for arginine, xyl = gene involved in xylose metabolism, ilv = gene coding for isoleucine and valine, leu = gene coding for leucine, pro = gene coding for proline, his = gene coding for histidine. Possible sources of errors in this experiment include the risk of dislodging the mating pairs by causing shaking or jarring whilst removing samples. This would have led to no further gene transfer in the bacteria concerned and hence errors to the results as growth may not have occurred at a time point in which it would have otherwise. The time at which the genes appeared to be transferred was determined by plating onto selective media. It was necessary to not only select for the recombinant (the recipient after it has received genetic material) but also against the donor. This is known as counter selection and here was carried out by including nalidixic acid in the medium as this prevents donor growth but not the recipient. Nalidixic acid will also prevent further gene transfer. If it was not included the donor strain would also grow on the selective media and all samples would show growth at all times due to the presence of the donor. The procedure and accuracy of results could be improved by using a quantitative approach in which the number of recombinant colonies are counted at each time point using a viable count. In conclusion the rough mapping of gene order along the E. coli chromosome can be determined qualitatively by physically interrupting mating pairs of a donor and recipient E. coli strain at certain intervals and noting growth levels after plating on selective media. This highlights the basic concepts and methods of conjugation and recombination in bacteria. 
"""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"2. Predicted generated: {generated}")

test_text = """Imagine a world where deadly diseases run rampant, infecting millions of people and causing widespread suffering. This bleak scenario is a chilling reality in countries without mandatory vaccination policies. Vaccines have been hailed as one of the greatest medical advancements in history, effectively reducing the prevalence of deadly diseases and saving countless lives. However, despite overwhelming evidence supporting their safety and efficacy, some individuals continue to oppose mandatory vaccination policies. In this essay, I will argue in favor of implementing mandatory vaccination policies based on data and expert opinions from the medical field.

Firstly, it is crucial to acknowledge the scientific consensus regarding the safety and effectiveness of vaccines. Numerous studies conducted by reputable institutions have shown that vaccines are a safe and proven method of preventing infectious diseases. For example, a study published in the New England Journal of Medicine found that childhood vaccines greatly reduce the risk of diseases such as measles, mumps, and rubella. This research provides solid evidence to support the use of vaccines in preventing the spread of infectious diseases.

Furthermore, vaccines not only protect individuals who receive them but also support a vital concept known as herd immunity. Herd immunity occurs when a large portion of a population is immunized, making it much more difficult for a disease to spread. This is particularly important for individuals who cannot receive vaccinations due to medical reasons. For example, people with weakened immune systems or severe allergies rely on the protection provided by herd immunity. By refusing to get vaccinated, individuals not only put themselves at risk but also jeopardize the health and safety of those who cannot receive vaccines.

Opponents of mandatory vaccination policies often argue that vaccines can have adverse side effects. While it is true that vaccines can cause mild side effects like soreness or fever, serious complications are incredibly rare. According to the World Health Organization (WHO), the benefits of vaccination far outweigh the risks. In fact, the United States Centers for Disease Control and Prevention (CDC) state that the chance of serious side effects from vaccines is less than the chance of serious complications from the diseases they prevent.

Additionally, it is vital to consider the ethical implications of mandatory vaccination policies. Vaccinating oneself and one's children is not only a matter of personal choice but also a responsibility towards the community. Infectious diseases can spread quickly, resulting in outbreaks and potentially overwhelming healthcare systems. By implementing mandatory vaccination policies, we can ensure the protection of vulnerable individuals and prevent the unnecessary suffering caused by entirely preventable diseases.

In conclusion, the scientific evidence supports the implementation of mandatory vaccination policies. Vaccines have been proven to be safe and effective in preventing the spread of infectious diseases. Furthermore, mandatory vaccination supports the concept of herd immunity, protecting those who cannot receive vaccines due to medical reasons. While opponents of mandatory vaccination policies raise concerns about side effects, the risks are minimal compared to the potential consequences of uncontrolled outbreaks. Ultimately, implementing mandatory vaccination policies is not only a logical choice backed by scientific evidence but also an ethical responsibility towards public health and the well-being of our communities."""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """Mandatory vaccination policies deserve serious consideration in today's world. Vaccinations have been proven to save lives and protect communities from preventable diseases. With the help of experts in the medical field and the wealth of data available, it is clear that the benefits of mandatory vaccinations outweigh any potential drawbacks.

Firstly, vaccination is essential for individual and community health. Vaccines have been instrumental in eradicating diseases such as smallpox and reducing the prevalence of others, such as polio and measles. According to the World Health Organization (WHO), vaccines prevent 2-3 million deaths annually. This statistic alone highlights the importance of ensuring that everyone, regardless of personal beliefs, receives the necessary vaccinations to protect themselves and others.

Secondly, mandatory vaccination policies are necessary to achieve herd immunity. Herd immunity occurs when a significant portion of the population is immune to a disease, making it difficult for the disease to spread. According to the Centers for Disease Control and Prevention (CDC), herd immunity is especially important in protecting those who cannot be immunized, such as infants or people with weakened immune systems. By requiring vaccinations for all, we can create a safer and healthier environment for these vulnerable members of society.

Opponents of mandatory vaccination policies argue that they infringe upon personal freedom and individual choice. While it is important to respect personal autonomy, public health must be prioritized. As Dr. Esther Choo, an emergency physician and professor at Oregon Health & Science University, states, "Individual choice cannot come at the expense of collective safety." When it comes to public health, individual choices can have significant consequences for the rest of the population.

Furthermore, misinformation surrounding vaccinations has led to hesitancy and refusal to vaccinate. Vaccine critics often rely on anecdotal evidence and misinformation found on the internet, rather than scientific research. It is crucial to combat this spread of misinformation by implementing mandatory vaccination policies that ensure accurate information and access to vaccinations for all.

In conclusion, the evidence in favor of mandatory vaccination policies is abundantly clear. Vaccinations save lives, prevent the spread of diseases, and protect vulnerable members of society. While personal freedom is important, public health should be paramount in decision-making. As future leaders, we must advocate for mandatory vaccinations in order to promote the well-being of individuals and the community as a whole."""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """Mandatory vaccination policies have been a subject of ongoing debate in recent years. While some argue that these policies infringe upon personal freedoms, the overwhelming evidence from medical experts supports their implementation. Mandatory vaccination policies are crucial for safeguarding public health, preventing the spread of deadly diseases, and protecting vulnerable populations.

First and foremost, mandatory vaccination policies are essential for maintaining public health. Vaccines have been proven to be highly effective in preventing the spread of infectious diseases. According to the Centers for Disease Control and Prevention (CDC), vaccines have significantly decreased the incidence of diseases such as measles, polio, and hepatitis B in the United States. By ensuring a high vaccination rate within a population, mandatory policies help create herd immunity, reducing the overall risk of disease transmission. This is especially important for individuals who cannot be vaccinated due to medical reasons, such as those with weakened immune systems or allergies to vaccine components.

Moreover, mandatory vaccination policies play a crucial role in preventing outbreaks of deadly diseases. The recent resurgence of measles cases in several countries serves as a stark reminder of the potential consequences of low vaccination rates. Measles, a highly contagious and potentially fatal disease, can easily spread in communities where vaccination rates are low. In fact, the World Health Organization (WHO) reports that measles cases have increased globally by 300% in recent years, largely due to undervaccination. By mandating vaccinations, governments can effectively limit the spread of diseases and protect their populations from needless suffering and loss of life.

Furthermore, mandatory vaccination policies are vital for protecting vulnerable populations. Infants, the elderly, and individuals with compromised immune systems are particularly susceptible to the complications and severe consequences of vaccine-preventable diseases. By ensuring high vaccination rates, governments can create a protective barrier around these vulnerable populations, reducing their risk of exposure to potentially life-threatening illnesses. In this way, mandating vaccines demonstrates society's commitment to safeguarding the health and well-being of all its members.

Critics of mandatory vaccination policies often argue that these policies infringe upon personal freedoms and parental rights. However, it is important to remember that individual choices and actions can have far-reaching consequences for others. The decision not to vaccinate not only places the unvaccinated individual at risk but also exposes the broader community to potential outbreaks. Personal freedoms must be balanced against the collective responsibility to protect public health. Mandatory vaccination policies strike this necessary balance by encouraging individual responsibility while prioritizing the common good.

In conclusion, mandatory vaccination policies are crucial for maintaining public health, preventing outbreaks of deadly diseases, and protecting vulnerable populations. The overwhelming consensus among medical experts supports these policies as an effective means of reducing disease transmission and ensuring the well-being of communities. While personal freedoms are important, they must be balanced with the collective responsibility to protect public health. By implementing and enforcing these policies, governments can make significant strides towards safeguarding the health and safety of their populations."""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """With the advancement of medical science, vaccines have played a vital role in preventing the spread of infectious diseases. However, the issue of mandatory vaccination policies has become a controversial topic. While some argue that individuals should have the freedom to decide whether or not to vaccinate themselves and their children, it is important to consider the data and expert opinions from the medical field that strongly support the implementation of mandatory vaccination policies.

One of the main arguments in favor of mandatory vaccination is the concept of herd immunity. When a significant percentage of a population is immunized against a particular disease, it creates a protective barrier that limits the spread of the disease. This is especially important for individuals who are unable to receive vaccines due to medical conditions. By vaccinating those who can, we are protecting vulnerable members of society who may be unable to defend themselves against infectious diseases.

Additionally, studies have shown that vaccines are highly effective in preventing the spread of diseases. For example, the measles vaccine has been proven to be 97% effective in preventing this highly contagious disease. By ensuring a high vaccination rate, we can effectively reduce the chances of outbreaks and epidemics.

Furthermore, experts in the medical field strongly support the implementation of mandatory vaccination policies. The Centers for Disease Control and Prevention (CDC) and the World Health Organization (WHO) recommend vaccination as one of the most effective ways to prevent the spread of infectious diseases. These organizations widely endorse and promote immunization as a public health measure.

In addition to the medical benefits, mandatory vaccination policies also have economic advantages. The cost of treating vaccine-preventable diseases is substantial, both for individuals and for the healthcare system as a whole. By implementing mandatory vaccination policies, we can reduce the economic burden associated with treating these diseases and allocate resources more effectively.

Despite the overwhelming evidence in favor of mandatory vaccination policies, some individuals argue that these policies infringe upon personal freedoms. While it is important to respect individual autonomy, it is equally important to consider the potential harm that can be caused by the spread of vaccine-preventable diseases. In cases where individual choices can potentially harm others, such as in the case of infectious diseases, it is necessary to prioritize the greater good.

In conclusion, mandatory vaccination policies are not only supported by data and expert opinions from the medical field, but they also promote the health and well-being of society as a whole. By ensuring a high vaccination rate, we can protect vulnerable members of society, prevent the spread of diseases, and reduce the economic burden associated with treating vaccine-preventable diseases. While individual freedoms are important, it is essential to consider the potential harm caused by the spread of infectious diseases and prioritize public health. Mandatory vaccination policies are a necessary step towards a healthier and safer society."""""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """In today's society, the issue of mandatory vaccination policies is a divisive topic that stirs up passionate debates. While some argue that individual liberties should take precedence, it is crucial to consider the overwhelming evidence and expert opinions from the medical field. This essay will present arguments in favor of mandatory vaccination policies, highlighting the benefits they provide for public health and the protection of vulnerable populations.

One of the primary reasons to support mandatory vaccination policies is the undeniable impact they have on preventing the spread of contagious diseases. Vaccines are scientifically proven to lower the risk of contracting and transmitting infections. Take, for example, the measles outbreak that occurred in the United States in 2019. According to the Centers for Disease Control and Prevention (CDC), nearly all reported cases were individuals who were unvaccinated. This outbreak could have been prevented if vaccinations were mandatory, effectively reducing the threat to public health.

Additionally, mandatory vaccination policies safeguard those who cannot receive vaccines for medical reasons. Individuals with weakened immune systems, such as cancer patients undergoing chemotherapy or infants who are too young to be vaccinated, heavily rely on the concept of herd immunity. When a significant portion of the population is immunized, it creates a protective barrier, preventing the spread of diseases to those who are susceptible. By mandating vaccinations, we ensure that these vulnerable populations are protected, drastically reducing the chances of severe illness or even death.

Moreover, it is important to note that the medical community overwhelmingly supports mandatory vaccination policies. Organizations such as the World Health Organization (WHO), the American Academy of Pediatrics (AAP), and the CDC all advocate for widespread vaccination programs. These institutions have spent years studying and compiling data that demonstrate the safety and effectiveness of vaccines. Therefore, their recommendations should carry considerable weight when considering public health policies.

Critics of mandatory vaccination policies often argue that they infringe upon individual freedoms. However, it is crucial to recognize that personal freedoms have limits when they potentially harm others. In the case of vaccinations, failing to vaccinate not only puts oneself at risk but also jeopardizes the well-being of the larger community. By imposing mandatory vaccinations, we strike a balance between individual autonomy and the common good, prioritizing the health of all.

In conclusion, mandatory vaccination policies have proven to be an effective method for preventing the spread of contagious diseases and protecting vulnerable populations. The overwhelming evidence and support from expert medical opinions highlight the necessity of these policies for public health. It is imperative that we prioritize the well-being of society as a whole over personal freedoms, as the consequences of vaccine-preventable infections can be severe and even deadly. By implementing and upholding mandatory vaccination policies, we can ensure a healthier and safer future for everyone."""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """My dear reader,

I, as a humble individual, strongly believe that mandatory vaccination policies are absolutely necessary to promote the well-being of our society and protect the health of individuals, especially the most vulnerable among us. By ensuring that everyone receives necessary vaccinations, we can significantly reduce the spread of dangerous pathogens and prevent the outbreak of devastating diseases.

Firstly, let us consider the overwhelming scientific evidence that supports the effectiveness of vaccines in preventing the spread of infectious diseases. Numerous studies conducted by renowned medical institutions have shown that vaccines are effective in preventing illness, hospitalization, and even death. For instance, a study published in the prestigious Journal of Infectious Diseases found that the measles vaccine prevented an estimated 31.8 million cases of measles and 4,200 deaths from the year 2000 to 2017 in the United States alone! The evidence is clear: vaccinations save lives and protect our communities.

Moreover, mandatory vaccination policies also help establish herd immunity, which is crucial in safeguarding the well-being of those individuals who cannot receive vaccines due to medical reasons. By ensuring that a significant proportion of the population is vaccinated, we create a protective barrier that prevents the easy transmission of diseases, effectively shielding those who are more susceptible to infection. This is especially important for individuals with compromised immune systems, infants too young to receive certain vaccines, and the elderly, who may be at greater risk of severe illness or complications. In essence, mandatory vaccination policies are a compassionate and necessary measure to protect the vulnerable members of our community.

However, there are those who argue against mandatory vaccination policies, claiming that they infringe upon personal freedoms and individual rights. While I understand the importance of personal autonomy, the health and safety of the community must also be considered. It is by balancing these rights and responsibilities that we can maintain a harmonious and healthy society. When one person chooses not to receive a vaccine, they not only put themselves at risk but also jeopardize the health of those around them. By making vaccination mandatory, we ensure that all individuals take responsibility for their role in preventing the spread of infectious diseases.

For this argument, it is only fitting to consider the opinion of esteemed medical experts. Dr. Anthony Fauci, Director of the U.S. National Institute of Allergy and Infectious Diseases, has stated, "Mandatory vaccination policies are crucial in protecting public health. They reduce the burden of disease and save countless lives." His expertise and experience leave no room for doubt regarding the efficacy of mandatory vaccination policies.

In conclusion, my dear reader, while we must respect personal freedoms and autonomy, we must also recognize the inherent responsibility we have to protect the health and well-being of our society. By implementing mandatory vaccination policies, we can effectively prevent the spread of dangerous diseases, save lives, and protect the vulnerable among us. It is crucial that we follow the guidance of medical experts and trust in the overwhelming evidence that supports the benefits of vaccination. Let us embrace this collective responsibility and work together to ensure a healthier future for all.

Yours faithfully,
[Your Name]"""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """Automation is when machines do work instead of people. It's like having a robot friend who can do lots of things for you! But some people are worried because they think that automation might take away their jobs. That means they might not have work to do anymore. It's important to understand what this could mean for our future job markets.

Economic forecasts can help us predict what might happen. One report from the World Economic Forum says that robots and computers could replace around 75 million jobs by the year 2022. That's a big number! But the same report also says that automation could create around 133 million new jobs. So, it's not all bad news! We might lose some jobs, but we could also gain new ones.

Technological reports give us more details about how automation is changing different industries. For example, in the manufacturing industry, many jobs are done by machines now. Robots can do repetitive tasks faster and more accurately than humans. This means that some people who used to work in factories might not have jobs anymore. But at the same time, new jobs are being created because someone needs to design, build, and fix these machines.

Automation is also affecting other fields. In the transportation industry, self-driving cars and trucks are becoming more common. This could mean fewer jobs for drivers. But it's also creating opportunities for people to work on developing and maintaining this technology. And in the customer service industry, chatbots and virtual assistants are helping customers without the need for as many human workers.

So, while automation might change the job market, it doesn't mean that there won't be any jobs left for humans. It will just be different jobs. As machines become more capable, it's important for us to learn new skills that machines can't do as well. This might mean focusing on things like creativity, problem-solving, and emotional intelligence. These are the skills that make us uniquely human!

In conclusion, automation is changing our job markets. Some jobs might be replaced by machines, but new jobs will also be created. Economic forecasts and technological reports show that there is still a place for humans in the future of work. We need to adapt and learn new skills to stay relevant. The future is exciting, and we can embrace the changes that automation brings!"""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """The futur is exciting and scarry at the same time. Technologie is advancing at a speedy pace, and automation is one of the big change that will impact the job markets in the future. Autamation is wen machines and robots do tasks that used to be done by peplo. On one hand, it can improve efficiency and saftey. But on the other hand, it can also lead to job loss and inequality. Economic forecasts and technological reports show both the positive and negative impact of automation on future job markets.

Lets strt with the positiv consekwences of autamation. Accrding to economic forecasts, it can increase productivity and save time. Robots can do repetative tasks faster and with fewer errors than humans. This can help companys produce more goods and services. For example, car manufacturers can use robots to build cars faster and more precisly. As a result, they can make more cars and provide them at a lower costs. This means companies can earn more mony, which can lead to encreased economic growf and better standard of living.

Technological reports also highlight how autamation can improve saftey. In dangerous jobs, like minning or construction, robots can do the work insted of humans. This reduces the risk of acciedents and injury. For instence, using robots to explore underwater or outer space can protect human lives. So autamation can save lives and make the world a safer place.

However, automation also has negativ impacts on job markets. Economic forecasts indicate that it can lead to job loss and unemployment. Wen robots do tasks that were previously done by humans, companies may not need as many workers. For example, cashier jobs in supermarkets may decres as self-checkout machines become more common. This can result in pepole losing there jobs and strugling to find new ones.

Technological reports also show how autamation can create inequality. Not all jobs are susceptibl to automation. Skilled jobs that requir creativity, critical thinking, and emotional intelligence are less likely to be replaced by machines. Jobs that requir social incteraction, like teaching or nursing, are also more resistnt to autamation. This means that pepole in these jobs may have beter job security and earn highr wages, while others workrs who lose there jobs to robots may have to accept lowr-paying jobs or undergo significnt retraining.

In concusin, the impact of autamation on future job markets can be both positiv and negativ. It can increase productivity, sav lives, and make work saftr. But it can also lead to job loss and inequlity. As we move forward into the futur, it is importnt to consider these effects and find ways to minimize the negativ impacts of autamation on workers."""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")

test_text = """The Impact of Automation on Future Job Markets

Automation is the process of using technology and machinery to perform tasks that were traditionally done by humans. While automation has been around for several decades, recent advancements in technology have accelerated its adoption across various industries. This has raised concerns about the impact of automation on future job markets. Economic forecasts and technological reports shed light on both the potential benefits and challenges associated with increasing automation.

According to economic forecasts, automation has the potential to significantly alter the job market in the coming years. The World Economic Forum estimates that by 2025, automation could displace 85 million jobs worldwide. This could lead to unemployment and job insecurity for many workers. Additionally, low-skilled jobs that are routine in nature, such as manufacturing and data entry, are at a higher risk of being automated. However, the same forecasts also highlight the creation of new jobs in emerging industries, such as artificial intelligence and robotics. These jobs require specialized skills and education, leading to a growing demand for workers with technical expertise.

Technological reports provide further insights into the impact of automation. For example, a report by the McKinsey Global Institute estimates that up to 45% of current work activities can be automated using existing technology. This indicates that automation has the potential to streamline processes and increase productivity. Industries such as transportation, retail, and food services are already experiencing automation through self-driving vehicles, cashier-less stores, and automated food delivery services. Such advancements can lead to cost savings for businesses and enhanced convenience for consumers.

However, the widespread adoption of automation also presents challenges. One concern is the displacement of workers who are ill-prepared for the changing job market. The World Economic Forum predicts that up to 97 million new roles may emerge that require a different skill set. This calls for investments in retraining and upskilling programs to help current workers transition into new roles. Additionally, automation can exacerbate income inequality. Skilled workers in technical fields may benefit from increased job opportunities and higher wages, while low-skilled workers may face limited job prospects and lower wages.

Despite the potential challenges, automation also offers opportunities for job creation and economic growth. The same McKinsey report mentioned earlier suggests that automation can contribute up to $15 trillion to the global economy by 2030. This is due to increased productivity and the development of new industries. Furthermore, automation can free up human workers from repetitive and mundane tasks, allowing them to focus on more complex and creative work. This can lead to more fulfilling jobs and higher job satisfaction.

In conclusion, automation is expected to have a significant impact on future job markets. Economic forecasts and technological reports indicate that while automation may lead to displacement and job insecurity, it also has the potential to create new jobs and enhance productivity. To mitigate the negative consequences, investments in retraining and upskilling programs are crucial. Additionally, policymakers should focus on ensuring that the benefits of automation are shared across all sectors of society. By carefully managing the transition to a more automated future, we can embrace the opportunities while minimizing the challenges posed by automation."""
generated = predict_generated(test_text, model, tokenizer, device, MAX_LENGTH)
print(f"1. Predicted generated: {generated}")
