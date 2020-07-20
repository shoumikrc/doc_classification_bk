# Black Knight document classification problem

## Problem statement
Black Knight operations involve processesing documents related to mortgages. They often only have access to a document we have is a scan of a fax of a print out of the document. Their system is able to read and comprehend that document, turning a PDF into structured business content that in turn their customers can act on.

The dataset provided represents the output of the OCR stage of the data pipeline. Due to sensitive nature of the financial documents the raw text has been obscured by mapping each word to a unique value. If the word appears in multiple documents then that value will appear multiple times. The word order for the dataset comes directly from the OCR layer, so it should be roughly in order.

## Objective:
1. Train a classification model

2. Deploy the trained model to a public cloud platform as a webservice


## Part 1 Machine Learning phase

The python notebook doc_classification_exploration.ipynb corresponds to the exploratory data analysis phase of the provided dataset and the benchmarking of different classification models. The objective is to understand the data and identify the best classification model to be deployed as a webservice. 

### Conclusionfrom exploration phase: 

Document category distribution

Benchmarking

Next based on the observations, LOGISTIC REGRESSION model with L2 regularization was selected to be deployed as a webservice for document classsification. It had the highest average test accuracy with good generalization. The notebook train_predict.ipynb trains a model and saves the trained model as a pickle file to be used later in the web deployement phase. The TF-IDF Vectorizer from the feature transformation phase is saved as vectorizer.pkl. This pickle files helps to load the learned TF_IDF matrix which shall be used to transform the query document during testing the document classifciation web tool.

To have an idea about the trained classification model's performance, the data was split to train/validation (90%/10%) using stratified sampling. The confusion matrix of the validation set is as follows.

## Part 2 Web deployment phase

