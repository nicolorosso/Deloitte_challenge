import io
import ast
import json
import pandas as pd
from moto import mock_s3, mock_kinesis
from src.dashboard_predictions.data_generator import data_classification_processor, classify_new_processed_observation
import streamlit as st


def lambda_handler_predictor(s3_client, destination_bucket, s3_resource, daily_transactions_bucket,
                             event={}, context={}):

    daily_transactions = []

    objects = s3_client.list_objects_v2(Bucket=destination_bucket)
    st.info(f"Lambda handler is processing transactions and running ML models...")
    for obj in objects['Contents']:
        transaction = s3_resource.Object(destination_bucket, obj['Key'])
        file_content = transaction.get()['Body'].read().decode('utf-8')
        transaction_dict = ast.literal_eval(file_content)
        observation_df = pd.DataFrame(transaction_dict)
        observation_df = data_classification_processor(observation_df)
        prediction = classify_new_processed_observation(observation_df)

        observation_df['Is_Fraud'] = prediction
        daily_transactions.append(observation_df)


    daily_transactions = pd.concat(daily_transactions)

    csv_buffer = io.BytesIO()
    daily_transactions.to_csv(csv_buffer)
    s3_resource.Object(daily_transactions_bucket, 'daily_transactions.csv').put(Body=csv_buffer.getvalue())






