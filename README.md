# Combating Fake News with Machine Learning

## 1. Introduction and Problem Statement

### 1.1. Context and Background Information
In today’s digital world, social media spreads information quickly, but it has also fueled the rise of fake news. Designed to mislead, fake news undermines trust and distorts our understanding of important events. Our project uses machine learning to distinguish real news from fake, helping to promote reliable media consumption.

### 1.2. Purpose and Objectives of the Project
Our goal is to create a machine learning model that accurately classifies news articles as real or fake, utilizing both Bag of Words and BERT embeddings for feature representation. By doing so, we aim to provide a tool that helps individuals verify information and make informed decisions. The key objectives of our project are:

    1. Determine whether we could accurately predict the likelihood of something being real or fake.
    
    2. Train a reliable model using both Bag of Words and BERT embeddings on diverse datasets.
    
      -  Evaluate the model’s accuracy and reliability across different contexts.
      -  Use this to determine whether BERT embeddings outperform Bag of Words in accuracy and effectiveness.
      
    3. To determine if past data can be used to predict whether a more recent news article is real or fake.
    
### 1.3. Problem Questions
This project explores the following research questions:
  - How can machine learning effectively differentiate between real and fake news articles?
  - Which features of news content are most indicative of authenticity or falsehood?
  - How can fake news detection models be optimized to reduce misclassification and enhance public trust in automated tools?
    
By addressing these questions, we aim to develop a scalable, evidence-based solution to combat misinformation and empower individuals to make informed decisions using credible sources!

## 2. Methodology and Data 

### 2.1. Theoretical Framework
This project is based on natural language processing (NLP) and machine learning. We used techniques like Bag of Words and BERT embeddings to represent text and applied models such as Fully Connected Neural Networks (FCNN) and Logistic Regression for classification. These methods help capture patterns in text data to accurately distinguish between real and fake news.

### 2.2. The project involves several stages:

#### 1. Data Collection and Preprocessing
We used datasets from platforms like Kaggle, covering general news, Twitter feeds, and specific topics like COVID-19. Preprocessing steps included text normalization, tokenization, and stopword removal to prepare the data for analysis.

    1. Dataset 1: Fake News Classification
         - (72,134 news articles;  35,028 real news; 37,106 fake news)
       
    2. Dataset 2: Twitter Dataset
         - (134,200 tweets, 68,900 real tweets, 65,300 fake tweets)

    3. Dataset 3: Covid 19 Fake News 
         - (8,560 news articles; 4,451 real news; 4,109 fake news)
  
    4. Dataset 4:  Fake News Detection
         - (44,898 news articles; 21,417 real news; 23,481 fake news)


#### 2. Model Development & Code Layout

    1. BOW - FCNN
        
        1. Loaded in the dataset (Kaggle)
        2. Delt with missing values/ preprocess text/ filtering (talked about later)
        3. TF-IDF/ tensor dataset
        4. Created neural network, trained, and evaluated it
        
            (lr= 0.001, hidden dim= 128, output dim = 2, 10 epochs)

    2. BOW - LR

        1. Loaded in the dataset (Kaggle)
        2. Delt with missing values/ preprocess text/ filtering (talked about later)
        3. TF-IDF/ tensor dataset
        4. Train model with LR

            LogisticRegression(max_iter=200, solver='saga', penalty='l2', C=1.0)
        
        5. Predicted on test set
        (single class case)


#### 3. Evaluation

We evaluated the reliability of our models by analyzing confusion matrices and comparing their accuracy scores. This allowed us to better understand the models’ performance, particularly their ability to correctly classify real and fake news while minimizing misclassifications.

## 3. Results & Discussion 

### 3.1 BOW 

1.  FCNN

<img width="871" alt="image" src="https://github.com/user-attachments/assets/cadb218b-42bd-46ed-92cb-81b77fbf1e5a">
 
2.  LR
    
<img width="871" alt="image" src="https://github.com/user-attachments/assets/e82a99c6-406e-405b-bb6c-2d71367c1664">

#### Thoughts

    - Extremely high accuracy raised flags
    - Looked into the data and realized the data sets were not as great as we thought…. Lots of duplicates, empty entries, etc..
    - Applied filters (min of 50 chars, not empty, not a duplicate) to combat this 
    - Accuracy was still extremely high…

### 3.2 BERT

1.  FCNN

<img width="433" alt="image" src="https://github.com/user-attachments/assets/4d07d4a7-8149-4892-91bf-4ca440a2561f">

2.  LR

<img width="433" alt="image" src="https://github.com/user-attachments/assets/12024e15-a3d8-41c4-9e4d-e9744bc1185d">

### 3.3 How It Went

#### What went right

1. Building a model using a Bag of words (FCNN and LR)
2. Preprocessing to combat issues with data
3. Testing model on varied datasets

#### Challenges

1. Data problems
    2.  Extracting the data
    3.  The data itself
4. Embedding issues
5. BERT 

    - Running BERT proved particularly challenging, and we were only able to complete it on the 2017 and Twitter datasets. 
      
### So...Can the BOW-FCNN predict the future?

#### Findings:

    - The only test we were able to complete involved evaluating whether our fake news classification model could predict outcomes on a COVID-19 dataset. We used the bag-of-words        method in combination with an FCNN model to train the classifier. However, the results showed only 50% accuracy, indicating that the model had no predictive power for             future scenarios.

<img width="652" alt="image" src="https://github.com/user-attachments/assets/f1a57560-77b2-4c9a-923a-8482a78b2063">
