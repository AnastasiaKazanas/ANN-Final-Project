# -*- coding: utf-8 -*-
"""Bert .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G_e5esbSCZVW9Nak7VyhUt7D_LuOHM1t
"""

import random
import torch
import random
import torch
import pickle
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

# !pip install tensorflow

#load data from kaggle

import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
import kagglehub
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_scheduler

# Download latest version
path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")

# Load the dataset
df = pd.read_csv("/root/.cache/kagglehub/datasets/saurabhshahane/fake-news-classification/versions/77/WELFake_Dataset.csv")

# Check the first few rows of the dataset
print(df.head())

#check what's missing in the data

# Check for missing values in the dataset
missing_values = df.isnull().sum()

# Check how many rows have missing data
rows_with_missing_values = df[df.isnull().any(axis=1)]

# Print out the missing values summary and rows with missing data
print("Missing values per column:\n", missing_values)
print("\nNumber of rows with missing values:", rows_with_missing_values.shape[0])

# Preview the rows with missing values
print("\nRows with missing values:")
print(rows_with_missing_values.head())

#clean the data of missing values

# Ensure that all values in 'text' are strings
df['text'] = df['text'].fillna('')  # Fill NaN values with an empty string
texts = df['text'].astype(str).tolist()  # Convert to list of strings

# # map labels to binary values
# df['label'] = df['label'].map({'true': 1, 'false': 0})

# Check the first few entries
print(texts[:1])

print(len(texts[0]))

# randomise the dataset
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     df['text'], df['label'], test_size=0.2, random_state=42
# )
print(dataset[0])
dataset = shuffle(dataset, random_state=42)
print(dataset[0])
#tokenize with bert
#mattias says to put this into batches (XXX - ?)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize the text, adjust the max_length parameter based on data analysis (XXX)
def encode_text(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='tf')

# Tokenize the text data
# train_encodings = encode_text(texts[:10])

"""

MAKE SURE THAT THE MAX LENGTH IS THE LONGEST TWEET SIZE FOR EACH FILE
MAKE A RESULT TABLE THAT SHOWS THE MAX_LENGTH OF EACH DATASET
MAX len

"""
train_encodings = tokenizer(list(train_texts), padding=True, truncation=True, max_length=512, return_tensors='pt')
test_encodings = tokenizer(list(test_texts), padding=True, truncation=True, max_length=512, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(list(train_labels))
test_labels = torch.tensor(list(test_labels))

# Check the tokenized data
print(train_encodings)

with open("Train_temp.pkl", "wb") as fOut:
    pickle.dump({'sentences': train_texts, 'embeddings': train_encodings},fOut)

with open("Test_temp.pkl", "wb") as fOut:
    pickle.dump({'sentences': test_labels, 'embeddings': test_encodings},fOut)

with open("Train_embeddings.pkl", "rb") as fOut:
  lol = pickle.load(fOut)
  print(lol)
