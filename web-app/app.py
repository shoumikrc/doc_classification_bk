"""
    File name: app.py
    Author: Shoumik Roychoudhury
    Date created: 6/19/2020
"""

import json
import boto3
import boto3.session
import pickle
from pre_trained_model import Classification


def handler(event, context):

    text = event['body'].strip('words\=')

    if(len(text.strip()) == 0):
        return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps({
            "prediction": "Empty Document",
            "confidence": "N.A."
        })
    }

    s3client = boto3.client('s3',aws_access_key_id='############',aws_secret_access_key='###########')
    model_obj = s3client.get_object(Bucket='doc-classification-model', Key='pretrained.pkl')
    vector_obj = s3client.get_object(Bucket='doc-classification-model', Key='vectorizer.pkl')

    model_body = model_obj['Body'].read()
    vector_body = vector_obj['Body'].read()

    predictor = Classification(model_body=model_body, vector_body=vector_body)
    prediction, confidence = predictor.predict(text)

    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps({
            "prediction": prediction,
            "confidence": confidence,
            "message": text
        })
    }
