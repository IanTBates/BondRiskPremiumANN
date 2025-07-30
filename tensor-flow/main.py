import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from keras import layers
import os

import tensorflow.python.keras

#Source py scripts
from data_processor import DataProcessor
from plot_maker import PlotMaker
from regressor_ann import ANN



# Activater environment: .\myenv\Scripts\activate
# Run with the following command: python tensor-flow/main.py

def main():

    #print(tf.__version__)
    #print(os.getcwd())

    dataframe = ""

    try:
        dataframe = pd.read_csv("../bloomberg_values.csv")
    except FileNotFoundError as e:
        print("File not found:", e)

    print(dataframe)

    targets = ["USGG3M Index", "USGG2YR Index", "USGG10YR Index", "USGG30YR Index"]
    predictors = ["SPX Index", "VIX Index", "GDP CQOQ Index", "CPI XYOY Index", "NAPMPMI Index", "IP  CHNG Index", "CPTICHNG Index", "USURTOT Index", "NFP TCH Index", "INJCJC Index", "LEI CHNG Index"]
    date_column = "Date"

    df = DataProcessor(dataframe)
    df.clean_na()
    df.set_date_index(date_column)
    df.set_predictors()
    df.set_target()

    df.get_target_set()
    df.get_preditor_set()

    

if __name__ == "__main__":
    main()
















