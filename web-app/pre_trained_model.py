"""
    File name: pre_trained_model.py
    Author: Shoumik Roychoudhury
    Date created: 6/17/2020
"""

import pickle as pk


class Classification:

    def __init__(self, model_body, vector_body):
        """
        Loads the pretrained machine learning model and vectorizer.
        Provides the predict method to predict label when a query is made.
        :param model_path: path to the machine learning model
        :param vectorizer_path: path to vectorizer
        """

        self.model = pk.loads(model_body)
        self.vectorizer = pk.loads(vector_body)
        # self.model = load(model_body)
        # self.vectorizer = load(vector_body)
        self.labels = [x.title() for x in
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

    def predict(self, words):
        """
        Predict the label and confidence given the words from a document.
        :param words: Words as a space separated string
        :return: predicted label and confidence in the prediction
        """
        pred = self.model.predict(self.vectorizer.transform([words]))
        conf = self.model.predict_proba(self.vectorizer.transform([words])).max()*100
        confidence = round(conf,2)
        return self.labels[pred[0]], confidence
