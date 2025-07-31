import shiny.time_series_regression as tsr
import pandas as pd

import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def main():
    data = pd.read_csv("bloomberg_values.csv")
    predictors = [data.drop(columns = "Date").columns]
    targets= ["USGG10YR Index"]

    eval_metrics = {
        "Mean Absolute Error": tf.keras.metrics.MeanAbsoluteError(),
        "Mean Absolute Percentage Effor":  tf.keras.metrics.MeanAbsolutePercentageError(),
        "Mean Squeared Error":  tf.keras.metrics.MeanSquaredError()
    }

    model_choices = {
        "Linear Model": tsr.get_linear_model(targets, predictors), 
        "Dense Model 1": tsr.get_dense_model1(targets, predictors), 
        "Dense Model 2": tsr.get_dense_model2(targets, predictors), 
        "Dense Model 3": tsr.get_dense_model3(targets, predictors), 
    }


    
    test_results = tsr.rolling_window(window_size=300, data=data, predictors = [data.drop( columns = "Date").columns], targets= ["USGG10YR Index"], model = model_choices["Dense Model 1"], eval_metrics= eval_metrics.values(), forecast_horizon= 5, date="Date")

if __name__ == "__main__":
    main()