import xgboost as xgb
from src.ml_algorithms.preprocessor.preprocessors import PreprocessDataFrame
from src.ml_algorithms.classification.classification_preprocessor import ClassificationDataframeProcessor
import vaex
import vaex.ml
import pandas as pd
import joblib
import pytest


def process_starting_df():
    df = pd.read_csv(r'C:\Users\mashfrog\PycharmProjects\DeloitteFinal\data\SupplyChainDataset (1).csv', encoding='latin-1')
    pdf = PreprocessDataFrame()
    df = pdf.add_target_variable(df)
    df = pdf.parse_column_names(df)
    df = pdf.drop_unuseful_columns(df)
    df = pdf.process_datetetimes(df)
    df = pdf.augment_with_network_features(df)
    return df


def generate_transaction(df):
    df = df.sample(1)
    new_raw_transaction = df
    transaction_id = df.index[0]
    return new_raw_transaction, transaction_id


def data_classification_processor(processed_df):
    new_observation = processed_df
    cdp = ClassificationDataframeProcessor()
    new_processed_observation = cdp.process_new_observation_for_classification(new_observation)
    new_processed_observation = cdp.cycletransform_month_day_weekday(new_processed_observation)
    new_processed_observation = cdp.return_new_processed_observation(new_processed_observation)
    return new_processed_observation



def classify_new_processed_observation(new_processed_observation):
    classifier_path = r'C:\Users\mashfrog\PycharmProjects\DeloitteFinal\src\ml_algorithms\serialized_models\classifier_xgb_serialized.json'
    loaded_xgb_classifier = xgb.XGBClassifier()
    loaded_xgb_classifier.load_model(classifier_path)
    prediction = loaded_xgb_classifier.predict(new_processed_observation)
    return prediction


