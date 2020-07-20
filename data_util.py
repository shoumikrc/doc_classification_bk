'''
 File name: data_util.py
 Author: Shoumik Roychoudhury
 Date created: 7/16/2020

'''

import pandas as pd
import numpy as np
import pickle as pk
from joblib import dump

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(filename):
    """
    Load the dataset and remove missing data
    filename:param
    data frame:return
    """
    # step 1: Load the data and include column headers
    df = pd.read_csv(filename)
    df.columns = ["label", "doc"]
    df = df.dropna()

    return df
    # step 2 Make a stratified sample of the classes and split into test and train sets

def create_train_test(df):
    """
      Create train test split from the incoming data frame.
      Create TFIDF feature
      data frame:param
      tfidf_train,tfidf_test,y_train,y_test:return
    """
    label_encoder=LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])


    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in skf.split(df['doc'], df['label']):
      strat_train_set = df.iloc[train_index]
      strat_test_set = df.iloc[test_index]

    strat_train_set = strat_train_set.dropna(subset=['doc'])
    strat_test_set = strat_test_set.dropna(subset=['doc'])

    #Training and test examples based on stratified indices
    x_train, y_train = strat_train_set['doc'], strat_train_set['label']
    x_test, y_test = strat_test_set['doc'], strat_test_set['label']
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.01, max_features=1037929, ngram_range=(1, 2))
    tfidf_train = vectorizer.fit_transform(x_train)

    #save learned TFIDF model as pickle file
    #dump(vectorizer, "vectorizer.joblib")
    pk.dump(vectorizer, open("vectorizer.pkl", 'wb'))

    tfidf_test = vectorizer.transform(x_test)

    return tfidf_train,tfidf_test,y_train,y_test
