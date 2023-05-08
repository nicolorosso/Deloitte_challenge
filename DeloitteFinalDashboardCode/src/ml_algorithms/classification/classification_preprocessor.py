import xgboost as xgb
import vaex
import vaex.ml
import pandas as pd
import joblib


class ClassificationDataframeProcessor:

    classifier_path = r'C:\Users\mashfrog\PycharmProjects\DeloitteFinal\src\ml_algorithms\serialized_models\classifier_xgb_serialized.json'
    loaded_xgb_classifier = xgb.XGBClassifier()
    loaded_xgb_classifier.load_model(classifier_path)

    def __init__(self):
        self.columns_for_ml = ['Type','Days_for_shipment_scheduled','Benefit_per_order',
                               'Sales_per_customer','Delivery_Status','Late_delivery_risk',
                               'Customer_Segment','Latitude','Longitude','Order_City',
                               'Order_Item_Discount','Order_Item_Discount_Rate',
                               'Order_Item_Product_Price','Order_Item_Profit_Ratio','Order_Item_Quantity',
                               'Sales','Order_Item_Total','Product_Category_Id', 'Product_Name',
                               'Shipping_Mode','Category_Department_Name','shipping_date_DateOrders',
                               'Source_Centrality','Order_Month','Order_Day','Order_Weekday']

        self.cols_for_xgb_classifier = self.loaded_xgb_classifier.get_booster().feature_names
        self.mhe = joblib.load(r'C:\Users\mashfrog\PycharmProjects\DeloitteFinal\src\ml_algorithms\serialized_models\serialized_vaex_mhe.joblib')
        self.ohe = joblib.load(r'C:\Users\mashfrog\PycharmProjects\DeloitteFinal\src\ml_algorithms\serialized_models\serialized_vaex_ohe.joblib')

    def process_new_observation_for_classification(self, new_observation):
        new_observation['shipping_date_DateOrders'] = pd.to_datetime(new_observation['shipping_date_DateOrders'], unit='ms')
        new_observation['order_date_DateOrders'] = pd.to_datetime(new_observation['order_date_DateOrders'], unit='ms')

        new_observation['Order_Month'] = new_observation.order_date_DateOrders.dt.month
        new_observation['Order_Day'] = new_observation.order_date_DateOrders.dt.day
        new_observation['Order_Weekday'] = new_observation.order_date_DateOrders.dt.dayofweek

        ml = new_observation[self.columns_for_ml]
        return ml

    def cycletransform_month_day_weekday(self,new_observation):
        ml = vaex.from_pandas(df=new_observation, copy_index=False)
        ml = vaex.ml.CycleTransformer(features=['Order_Month'], n=12).fit_transform(ml)
        ml = vaex.ml.CycleTransformer(features=['Order_Day'], n=31).fit_transform(ml)
        ml = vaex.ml.CycleTransformer(features=['Order_Weekday'], n=7).fit_transform(ml)
        ml = ml.to_pandas_df()
        return ml


    def fix_observation_cols_for_prediction(self,new_observation):
        cols_for_xgb_classifier = self.cols_for_xgb_classifier
        cols_to_add = set(cols_for_xgb_classifier).difference(set(new_observation.columns))
        cols_to_remove = set(new_observation.columns).difference(set(cols_for_xgb_classifier))

        if len(cols_to_add) != 0:
            new_observation[list(cols_to_add)] = 0
        if len(cols_to_remove) != 0:
            new_observation = new_observation.drop(list(cols_to_remove), axis=1)

        return new_observation[cols_for_xgb_classifier]


    def return_new_processed_observation(self,new_observation):
        ohe = self.ohe
        mhe = self.mhe

        X_new = new_observation.drop(['Order_Month','Order_Day','Order_Weekday'],axis=1)
        X_new = vaex.from_pandas(df=X_new, copy_index=False)

        X_new = ohe.transform(X_new)
        X_new = mhe.transform(X_new)

        X_new = X_new.to_pandas_df()

        X_new = self.fix_observation_cols_for_prediction(new_observation)

        return X_new




