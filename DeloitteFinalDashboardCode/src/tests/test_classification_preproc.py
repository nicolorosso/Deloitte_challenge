from ..ml_algorithms.preprocessor.preprocessors import PreprocessDataFrame
from ..ml_algorithms.classification.classification_preprocessor import ClassificationDataframeProcessor
import vaex
import vaex.ml
import pandas as pd
import joblib
import pytest


@pytest.fixture
def process_starting_df():
    df = pd.read_csv(r'SupplyChainDataset (1).csv', encoding='latin-1')
    pdf = PreprocessDataFrame()
    df = pdf.add_target_variable(df)
    df = pdf.parse_column_names(df)
    df = pdf.drop_unuseful_columns(df)
    df = pdf.process_datetetimes(df)
    df = pdf.augment_with_network_features(df)
    return df


def test_classification_processor(process_starting_df):
    new_observation = process_starting_df.loc[10]
    new_observation = pd.DataFrame(new_observation).T
    cdp = ClassificationDataframeProcessor()
    new_observation = cdp.process_new_observation_for_classification(new_observation)
    new_observation = cdp.cycletransform_month_day_weekday(new_observation)
    new_observation = cdp.return_new_processed_observation(new_observation)
    assert new_observation.columns.shape[0] == 56










