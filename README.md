
# Combating Fake News with Machine Learning - Rohan, Sasha, Anastasia

## 1. Introduction and Problem Statement


### 1.1. Context and Background Information

In today’s digital world, social media has made it easier than ever to share information, but it has also amplified the spread of fake news. This misinformation undermines trust in media and makes it harder for individuals to determine which sources are reliable. Our project tackles this issue by using machine learning techniques to classify news articles and social media posts as real or fake. By doing so, we aim to promote more reliable media consumption and combat the spread of misinformation.

**As seen below...**

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/user-attachments/assets/cfb54d1c-76c6-42c7-850b-36d141601a41" alt="Image 1" width="500"/>
  <img src="https://github.com/user-attachments/assets/0bce7500-8ccf-4e42-bab7-fff2798e150e" alt="Image 2" width="500"/>
</div>


### 1.2. Purpose and Objectives of the Project

Our goal is to create a machine learning model capable of accurately classifying news articles and social media posts as real or fake. We aim to:

1. Determine whether we could accurately predict the likelihood of something being real or fake.

2. Train a reliable model using both Bag of Words and BERT embeddings on diverse datasets.
   - Evaluate the model’s accuracy and reliability across different contexts.
   - Use this to determine whether BERT embeddings outperform Bag of Words in accuracy and effectiveness.

3. Determine if past data can be used to predict whether a more recent news article is real or fake.

    
### 1.3. Problem Questions

This project explores the following research questions:
  - How can machine learning effectively differentiate between real and fake news articles?
  - Which features of news content are most indicative of authenticity or falsehood?
  - How can fake news detection models be optimized to reduce misclassification and enhance public trust in automated tools?
    
By addressing these questions, we aim to develop a scalable, evidence-based solution to combat misinformation and empower individuals to make informed decisions using credible sources!


## 2. Methodology and Data 


### 2.1. What We Created

1. **Bag of Words (BOW)**:
   
   - Developed models using a BOW representation, which was processed through:
     - Fully Connected Neural Networks (FCNN).
     - Logistic Regression (LR).  
   - We tested these models on four datasets, creating a total of eight configurations.

3. **BERT Embeddings**:
   
   - Replaced BOW with BERT embeddings to capture the contextual meaning of words.  
   - Used these embeddings with FCNN and LR models, applied to two datasets, creating four additional configurations.


### 2.2. Datasets

We used the following datasets from Kaggle:

1. **Fake News Classification**
   
   - 72,134 articles (35,028 real, 37,106 fake).  
   - [Dataset Link](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)
     
3. **Twitter Dataset**
   
   - 134,200 tweets (68,900 real, 65,300 fake).  
   - [Dataset Link](https://www.kaggle.com/datasets/sudishbasnet/truthseekertwitterdataset2023/data)
     
5. **COVID-19 Fake News Dataset**
   
   - 8,560 articles (4,451 real, 4,109 fake).  
   - [Dataset Link](https://www.kaggle.com/datasets/invalizare/covid-19-fake-news-dataset)
     
7. **Fake News Detection**
   
   - 44,898 articles (21,417 real, 23,481 fake).  
   - [Dataset Link](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets/data)


### 2.3. Code Layout

1. **Bag of Words - FCNN**:
   
   - Load and preprocess datasets (e.g., handle missing values, remove duplicates).  
   - Apply TF-IDF and create a tensor dataset.  
   - Train and evaluate a Fully Connected Neural Network (10 epochs, learning rate 0.001).  

3. **Bag of Words - LR**:
   
   - Load and preprocess datasets.  
   - Use TF-IDF to represent data.  
   - Train a Logistic Regression model and predict on the test set.

5. **BERT Models**:
   
   - Preprocess datasets and generate embeddings using BERT.  
   - Train models using FCNN and Logistic Regression.  


## 3. Results

### 3.1. Accuracy Metrics

1. **Bag of Words - FCNN**:  
   - Fake News Classification: 97%  
   - Twitter Dataset: 98%  
   - COVID-19 Fake News: 89%  
   - Fake News Detection: 99%  

2. **Bag of Words - Logistic Regression**:
   
   - Fake News Classification: 96%  
   - Twitter Dataset: 97%  
   - COVID-19 Fake News: 89%  
   - Fake News Detection: 99%  

4. **BERT Models**:
   
   - Fake News Classification (FCNN): 58%  
   - Twitter Dataset (FCNN): 51%  
   - Fake News Classification (LR): 65%  
   - Twitter Dataset (LR): 55%  


### 3.2. Challenges

1. **Data Issues**:
  
  - Datasets contained duplicates, missing values, and empty entries.  
  - We applied filters (e.g., minimum character count, removing duplicates) to improve quality.  

2. **Embedding Issues**:
  
  - Computational limits restricted BERT to only two datasets (Fake News Classification and Twitter).
    

## 4. Discussion

- Our BOW models performed well, achieving high accuracy on all datasets.  
- BERT embeddings, while theoretically more robust, yielded lower accuracy due to limited computational resources and smaller dataset coverage.
- The BOW-FCNN model’s low predictive accuracy of 50% on the COVID-19 dataset makes sense for several reasons:
  
    1. **Novelty of the Topic**:
       
    The COVID-19 dataset includes news articles about a specific, unprecedented global event that emerged after the training data was collected. The vocabulary, phrasing, and         context of COVID-related articles may not have been present in the datasets used for training, causing the model to struggle when generalizing to this new topic.
    	
     2.	**Lack of Contextual Understanding**:
        
    Bag of Words models represent text as a collection of word counts without capturing the context or relationships between words. For a nuanced and complex topic like COVID-19,     context is crucial for distinguishing between real and fake news, which BOW cannot handle effectively.
    
### 4.1 Insight About Logistic Regression Performing as Well as FCNN

Interestingly, Logistic Regression performed almost as well as FCNN in most scenarios, suggesting that certain individual words or phrases carry significant predictive power for distinguishing real from fake news. This observation highlights a few key points:

1.	**High-Impact Features**:
 
Logistic Regression assigns weights to individual words, and the success of the model indicates that some words or phrases—like “breaking,” “exclusive,” or “official”—            might strongly correlate with fake or real news. These high-weighted features drive the model’s predictive power, even without complex representations.
    
2.	**Simplicity in Linear Models**:
 
Logistic Regression’s strong performance shows that for this task, the relationships between features and labels might be relatively linear. In such cases, FCNNs, which are       designed to capture complex, nonlinear patterns, may not offer significant advantages.
    
## 5. Conclusion

This project demonstrated the effectiveness of machine learning models in distinguishing real and fake news. While BOW models performed consistently well, challenges with data quality and computational limitations highlighted areas for improvement. Future work could focus on refining preprocessing techniques, expanding dataset coverage, and leveraging more computational resources for BERT models.

**Presentation Link**: https://docs.google.com/presentation/d/1G1oTerb_HOjSipo4u55pR_1wD2zERMPVxJUTXobLqfo/edit?usp=sharing
