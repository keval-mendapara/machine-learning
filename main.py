import numpy as np
import pandas as pd
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load and preprocess data
emails = pd.read_csv('emails.csv')
emails.dropna(subset=['email_content', 'category'], inplace=True)
print(f"Loaded {len(emails)} emails.")

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

emails['sentiment'] = emails['email_content'].apply(get_sentiment)
print(emails['sentiment'].value_counts())

# Plot the sentiment distribution
emails['sentiment'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Number of Emails')
plt.title('Sentiment Analysis of Emails')
plt.show()

# Text Preprocessing for Classification
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

emails['processed_content'] = emails['email_content'].apply(preprocess_text)

# Email Classification with Logistic Regression
categories = ['Business', 'Personal', 'Spam']
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(emails['processed_content'])
y = emails['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Define the expected labels
present_labels = sorted(list(set(y_train).union(set(y_test))))
present_target_names = present_labels  # Since labels are already strings corresponding to categories

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=present_target_names, labels=present_labels))

# Advanced Classification with BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))

# Prepare data for BERT
train_texts = emails.loc[X_train.indices, 'processed_content'].tolist()
test_texts = emails.loc[X_test.indices, 'processed_content'].tolist()
train_labels = y_train.apply(lambda x: categories.index(x)).tolist()
test_labels = y_test.apply(lambda x: categories.index(x)).tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmailDataset(train_encodings, train_labels)
test_dataset = EmailDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.evaluate()
