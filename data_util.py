'''
 File name: data_util.py
 Author: Shoumik Roychoudhury
 Date created: 7/16/2020

'''

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from joblib import dump

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

category_ids = [x.title() for x in
                       ['APPLICATION',
                        'BILL',
                        'BILL BINDER',
                        'BINDER',
                        'CANCELLATION NOTICE',
                        'CHANGE ENDORSEMENT',
                        'DECLARATION',
                        'DELETION OF INTEREST',
                        'EXPIRATION NOTICE',
                        'INTENT TO CANCEL NOTICE',
                        'NON-RENEWAL NOTICE',
                        'POLICY CHANGE',
                        'REINSTATEMENT NOTICE',
                        'RETURNED CHECK']
                       ]
        

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
    
    features = vectorizer.get_feature_names()
    dfs = top_feats_by_class(tfidf_train, y_train, features, min_tfidf=0.1, top_n=25)
    plot_tfidf_classfeats_h(dfs)
    
    

    #save learned TFIDF model as pickle file
    #dump(vectorizer, "vectorizer.joblib")
    pk.dump(vectorizer, open("vectorizer.pkl", 'wb'))

    tfidf_test = vectorizer.transform(x_test)

    return tfidf_train,tfidf_test,y_train,y_test


def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    ''' Return a list of dfs, where each df holds top_n features and their mean tfidf value
        calculated across documents with the same class label. '''
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs


def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(9, 70), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(len(dfs),1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        #ax.set_xlabel("Mean Tf-Idf Score", fontsize=14)
        ax.set_title("label = " + str(category_ids[df.label]), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        #plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
        
    plt.show()


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df