# Combating Fake News with Machine Learning
## 1 - Introduction and Problem Statement
### 1.1 Context and Background Information
In today’s digital world, social media spreads information quickly, but it has also fueled the rise of fake news. Designed to mislead, fake news undermines trust and distorts our understanding of important events. Our project uses machine learning to distinguish real news from fake, helping to promote reliable media consumption.
### 1.2 Purpose and Objectives of the Project
Our goal is to create a machine learning model that accurately classifies news articles as real or fake, utilizing both Bag of Words and BERT embeddings for feature representation. By doing so, we aim to provide a tool that helps individuals verify information and make informed decisions. The key objectives of our project are:
  - Train a reliable model using both Bag of Words and BERT embeddings on diverse datasets, including a 2017 news article dataset and a Twitter dataset.
  - Evaluate the model’s accuracy and reliability across different contexts.
### 1.3 Problem Questions
This project explores the following research questions:
  - How can machine learning effectively differentiate between real and fake news articles?
  - Which features of news content are most indicative of authenticity or falsehood?
  - How can fake news detection models be optimized to reduce misclassification and enhance public trust in automated tools?
By addressing these questions, we aim to develop a scalable, evidence-based solution to combat misinformation and empower individuals to make informed decisions using credible sources.

## 2 - Methodology and Data 
### 2.1 Theoretical Framework
This project is based on natural language processing (NLP) and machine learning. We used techniques like Bag of Words and BERT embeddings to represent text and applied models such as Fully Connected Neural Networks (FCNN) and Logistic Regression for classification. These methods help capture patterns in text data to accurately distinguish between real and fake news.
### 2.2 Project Design
The project involves several stages:
#### 1. Data Collection and Preprocessing
We used datasets from platforms like Kaggle, covering general news, Twitter feeds, and specific topics like COVID-19. Preprocessing steps included text normalization, tokenization, and stopword removal to prepare the data for analysis.
#### 2. Model Development
We began with simpler models for baseline performance, such as Logistic Regression, and advanced to more complex approaches like Fully Connected Neural Networks (FCNN). Feature extraction techniques included Bag of Words and BERT embeddings.
#### 3. Model Validation
To ensure robust performance, we used cross-validation and tested the models on various datasets to confirm consistency and adaptability.
#### 4. Resources and Data
We sourced our datasets from reputable platforms like Kaggle, encompassing various types of news contexts including general news, Twitter feeds, and COVID-19 related news. The data preprocessing phase involved normalizing text, removing noise, and employing techniques like tokenization and stopwords removal.
#### 5. Evaluation
We evaluated model reliability through cross-validation and extensive testing across datasets. This process reduced overfitting and improved generalizability, ensuring the models are effective in real-world scenarios.

## 3 - Results & Discussion 
### 3.1 Context of the Results

We achieved promising results, with our models demonstrating high accuracy rates in distinguishing between real and fake news. The models performed well across various contexts, including newer topics incorporated post-training (for Bag of Words).

Our results show that the models are capable of learning and generalizing from the training data to effectively classify new, unseen data. This suggests that with continuous updates and training, such models can remain effective as the landscape of news evolves.

### 3.2 Challenges Encountered
Running BERT proved particularly challenging, and we were only able to complete it on the 2017 and Twitter datasets. Additionally, our datasets contained many duplicate entries, making the data cleaning process more complex and time-consuming. To address these challenges, we refined our models and dedicated significant effort to thorough data preprocessing.
### 3.3 Relation to Other Works
This project builds on prior research in NLP and fake news detection, extending it by utilizing both traditional methods like Bag of Words and advanced techniques such as BERT embeddings. It further innovates by exploring newer neural network architectures and evaluating model adaptability to evolving topics, including COVID-19.

## 4 - Conclusion
The success of our project indicates that machine learning can be an effective tool against the spread of misinformation. By continuing to refine our models and expand their capabilities, we aim to create a reliable tool that can play a crucial role in the fight for truth in media.
