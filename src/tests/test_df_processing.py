import numpy as np
import pytest
import pandas as pd
import networkx as nx
from ..ml_algorithms.preprocessor.preprocessors import PreprocessDataFrame


def test_complete_processing():
    df_processed = pd.read_json(r'dataset.json')
    df_original = pd.read_csv(r'SupplyChainDataset (1).csv',encoding='latin-1')

    preprocessor = PreprocessDataFrame(df_original).add_target_variable()
    preprocessor = PreprocessDataFrame(preprocessor).parse_column_names()
    preprocessor = PreprocessDataFrame(preprocessor).drop_unuseful_columns()
    preprocessor = PreprocessDataFrame(preprocessor).process_datetetimes()
    preprocessor = PreprocessDataFrame(preprocessor).augment_with_network_features()


    assert (df_processed['Days_for_shipping_real'] == preprocessor['Days_for_shipping_real']).all()