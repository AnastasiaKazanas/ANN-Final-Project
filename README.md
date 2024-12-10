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

### 2.2. Project Design
The project involves several stages:

#### 1. Data Collection and Preprocessing
We used datasets from platforms like Kaggle, covering general news, Twitter feeds, and specific topics like COVID-19. Preprocessing steps included text normalization, tokenization, and stopword removal to prepare the data for analysis.

 - Dataset 1: Fake News Classification
     - (72,134 news articles;  35,028 real news; 37,106 fake news)
       
 - Dataset 2: Twitter Dataset
     - (134,200 tweets, 68,900 real tweets, 65,300 fake tweets)

 - Dataset 3: Covid 19 Fake News 
     - (8,560 news articles; 4,451 real news; 4,109 fake news)
  
 - Dataset 4:  Fake News Detection
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


#### 5. Evaluation
We evaluated the reliability of our models by analyzing confusion matrices and comparing their accuracy scores. This allowed us to better understand the models’ performance, particularly their ability to correctly classify real and fake news while minimizing misclassifications.

## 3. - Results & Discussion 
### 3.1 Context of the Results

We achieved promising results, with our models demonstrating high accuracy rates in distinguishing between real and fake news. The models performed well across various contexts, including newer topics incorporated post-training (for Bag of Words).

Our results show that the models are capable of learning and generalizing from the training data to effectively classify new, unseen data. This suggests that with continuous updates and training, such models can remain effective as the landscape of news evolves.

### 3.2 Challenges Encountered
Running BERT proved particularly challenging, and we were only able to complete it on the 2017 and Twitter datasets. Additionally, our datasets contained many duplicate entries, making the data cleaning process more complex and time-consuming. To address these challenges, we refined our models and dedicated significant effort to thorough data preprocessing.
### 3.3 Relation to Other Works
This project builds on prior research in NLP and fake news detection, extending it by utilizing both traditional methods like Bag of Words and advanced techniques such as BERT embeddings. It further innovates by exploring newer neural network architectures and evaluating model adaptability to evolving topics, including COVID-19.

## 4. - Conclusion
The success of our project indicates that machine learning can be an effective tool against the spread of misinformation. By continuing to refine our models and expand their capabilities, we aim to create a reliable tool that can play a crucial role in the fight for truth in media.
