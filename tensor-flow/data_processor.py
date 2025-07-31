import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, raw_dataframe: pd.DataFrame):
        self.raw_dataframe = raw_dataframe
        self.dataframe = self.clean_na(self.raw_dataframe)
        self.index = ""
        self.predictors = []
        self.targets = []

    def set_date_index(self, index):
        self.index = index
        #self.dataframe[index] = pd.to_datetime(self.dataframe[index], format='%m/%d/%Y')
        self.dataframe.set_index(index, inplace=True)

    def get_target_set(self):
        target_set = self.dataframe[self.targets]
        return target_set
    
    def get_predictor_set(self):
        predictor_set = self.dataframe[self.predictors]
        return predictor_set

    def set_targets(self, targets):
        self.targets = targets

    def set_predictors(self, predictors):
        self.predictors = predictors

    def create_lags(self, df, column, lags):
        """
        This function creates the lags of the target columns        
        
        """
        if isinstance(lags, int):
            lag_list = range(1, lags + 1)
        else:
            lag_list = lags

        for lag in lag_list:
         df[f"{column}_lag{lag}"] = df[column].shift(lag)

        return df

    def get_dataframe(self):
        return self.dataframe


    def clean_na(self, dataframe: pd.DataFrame):
        """
        This function cleans the datafram
        """
        na_sum = dataframe.isna().sum()
        print(na_sum)

        clean_dataframe = dataframe.dropna()

        return clean_dataframe


    def transform():
        pass



