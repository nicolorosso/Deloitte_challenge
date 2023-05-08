import altair as alt
import io
import ast
import pandas as pd
import json
import boto3
import random
import string
import time
from moto import mock_kinesis, mock_s3
from data_generator import generate_transaction, process_starting_df
from src.mocked_aws_environment.base_mocked import lambda_handler_predictor
import streamlit as st


# Producer function to produce random data and put it into the Kinesis stream
def produce_data(kinesis_client, stream_name):
    for i in range(15):
        transaction, transaction_id = generate_transaction(df)
        print(f"Producing data: {transaction_id}")
        kinesis_client.put_record(
            StreamName=stream_name,
            Data=json.dumps(transaction.to_dict()),
            PartitionKey='1'
        )
        time.sleep(1)


# Consumer function to consume data from the Kinesis stream
def consume_data(kinesis_client, stream_name, s3_client, destination_bucket):
    response = kinesis_client.describe_stream(StreamName=stream_name)
    shard_id = response['StreamDescription']['Shards'][0]['ShardId']
    shard_iterator_response = kinesis_client.get_shard_iterator(
        StreamName=stream_name,
        ShardId=shard_id,
        ShardIteratorType='LATEST'
    )
    shard_iterator = shard_iterator_response['ShardIterator']

    for i in range(15):
        response = kinesis_client.get_records(ShardIterator=shard_iterator)
        for record in response['Records']:
            s3_key = f"{shard_id}_{record['SequenceNumber']}.json"
            s3_client.put_object(Bucket=destination_bucket, Key=s3_key, Body=record['Data'])
            print(f'Shard {s3_key} successfully uploaded to the S3 Bucket')

        shard_iterator = response['NextShardIterator']
        time.sleep(1)


# Main function to run the producer and consumer functions
def main():
    with mock_kinesis(), mock_s3():
        # Create a Kinesis client and a S3 client to store data
        kinesis_client = boto3.client('kinesis', region_name='us-east-1')
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_resource = boto3.resource('s3', region_name='us-east-1')

        # Create a Kinesis stream
        stream_name = 'test_stream'
        kinesis_client.create_stream(StreamName=stream_name, ShardCount=1)

        # Create an S3 bucket
        destination_bucket = 'destinationBucket'
        s3_client.create_bucket(Bucket=destination_bucket)
        daily_transactions_bucket = 'dailyTransactions'
        s3_client.create_bucket(Bucket=daily_transactions_bucket)

        # Run the producer and consumer functions
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as executor:
                executor.submit(produce_data, kinesis_client, stream_name)
                executor.submit(consume_data, kinesis_client, stream_name, s3_client, destination_bucket)

        except KeyboardInterrupt:
            print("Stopping...")

        st.info('Working day over... retrieving data from the bucket')

        lambda_handler_predictor(s3_client=s3_client,
                                 destination_bucket=destination_bucket,
                                 s3_resource=s3_resource,
                                 daily_transactions_bucket=daily_transactions_bucket)


        todays_transactions_df = []
        todays_transactions_list = s3_client.list_objects_v2(Bucket=daily_transactions_bucket)
        print(f'FTRNSAZIONI DI OGGI: {todays_transactions_list}')
        for obj in todays_transactions_list['Contents']:
            transaction = s3_resource.Object(daily_transactions_bucket, obj['Key'])
            file_content = transaction.get()['Body'].read()
            df = pd.read_csv(io.BytesIO(file_content))
            todays_transactions_df.append(df)
        todays_transactions_df = pd.concat(todays_transactions_df)
        return todays_transactions_df


def run_app():
    st.title("Transactions KPI Dashboard")

    if st.button("Retrieve Today's Transactions"):
        df = main()
        df = df.drop('Unnamed: 0', axis=1)
        st.dataframe(df)

        st.divider()

        col1, col2 = st.columns(2)


        with col1:
            st.subheader('Order Item Quantity distribution')
            st.altair_chart(alt.Chart(df).mark_bar().encode(
                alt.X("Order_Item_Quantity:Q", bin=True),
                y='count()'
            ).interactive())  # aggiunto .interactive()
        with col2:
            st.subheader('Order Item Product Price distribution')
            st.altair_chart(alt.Chart(df).mark_bar().encode(
                alt.X("Order_Item_Product_Price:Q", bin=True),
                y='count()'
            ).interactive())  # aggiunto .interactive()

        st.write(f"Biggest order contained {df.Order_Item_Quantity.max()} items!")
        st.write(f"Most expensive item bought was {round(df.Order_Item_Product_Price.max())}$!")
        st.write(f"A total of {round(df.Is_Fraud.sum())} orders were predicted as fraud!")
        try:
            frauds_list = df.loc[df['Is_Fraud']==1].index.tolist()
            for fraud_id in frauds_list:
                print(f"Refer to shardId-000000000000_{fraud_id}.json")
        except:pass

        st.divider()

        st.subheader('Most productive stores today')
        df = df.rename(columns={'Latitude':'LAT', 'Longitude':'LON'})
        st.map(df[['LAT', 'LON']])


if __name__ == "__main__":
    df = process_starting_df()
    run_app()
