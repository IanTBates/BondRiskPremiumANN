import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from keras import layers
import os

#Source py scripts
import data_processor
import plot_maker
import regressor_ann



# Activater environment: .\myenv\Scripts\activate
# Run with the following command: python tensor-flow/main.py

def main():

    #print(tf.__version__)
    #print(os.getcwd())

    dataframe = ""

    try:
        dataframe = pd.read_csv("./bloomberg_values.csv")
    except FileNotFoundError as e:
        print("File not found:", e)

    print(dataframe)




    

if __name__ == "__main__":
    main()
















