import numpy as np
import networkx as nx
import pandas as pd


class PreprocessDataFrame:
    def __init__(self):
        self.to_drop = ['Customer_City', 'Customer_Country', 'Customer_Email', 'Customer_Fname', 'Customer_Id',
                   'Customer_Lname', 'Customer_Password', 'Customer_Street', 'Customer_Zipcode',
                   'Product_Description', 'Product_Image', 'Product_Status', 'Order_Zipcode', 'Order_Item_Id',
                   'Order_Country', 'Order_Customer_Id', 'Order_Country', 'Order_Region', 'Order_State',
                   'Market', 'Customer_State', 'Order_Id', 'Order_Item_Cardprod_Id', 'Product_Card_Id',
                   'Order_Profit_Per_Order', 'Department_Id', 'Product_Price', 'Category_Id']

    def add_target_variable(self, df):
        df['Is_Fraud'] = df['Order Status'].apply(lambda x: 1 if x=='SUSPECTED_FRAUD' else 0)
        return df

    def parse_column_names(self, df):
        df.columns = [i.replace(' ','_').replace('(','').replace(')','') for i in df.columns]
        return df

    def drop_unuseful_columns(self, df):
        to_drop = self.to_drop

        df['Category_Department_Name'] = (df['Category_Name'] + '_' + df['Department_Name']) \
            .str.replace(' ', '_')

        df = df.drop(to_drop + ['Category_Name','Department_Name'], axis=1)
        return df

    def process_datetetimes(self, df):
        df.shipping_date_DateOrders = pd.to_datetime(df.shipping_date_DateOrders)
        df.shipping_date_DateOrders = (df.shipping_date_DateOrders.astype(np.int64) / int(1e6)).astype('int64')

        df.order_date_DateOrders = pd.to_datetime(df.order_date_DateOrders)
        df.order_date_DateOrders = (df.order_date_DateOrders.astype(np.int64) / int(1e6)).astype('int64')
        return df

    def augment_with_network_features(self, df):
        df['Source'] = ('LA' + df['Latitude'].astype('str') + '-' + 'LO' + df['Longitude'].astype('str'))
        G = nx.from_pandas_edgelist(df, source='Source', target='Order_City')
        df['Source_Centrality'] = df.Source.map(dict(G.degree))
        return df

    


