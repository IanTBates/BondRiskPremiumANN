import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Models
def get_linear_model(predictors, targets):
    linear_model = tf.keras.Sequential([
        layers.Dense(units=1)
    ])

def get_dense_model1(predictors, targets):
    dense_model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10, activation='relu', input_shape=(len(predictors),)),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=len(targets))
    ])

def get_dense_model2(predictors, targets):
    dense_model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(units=60, activation='relu', input_shape=(len(predictors),)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=60, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=len(targets))
    ])

def get_dense_model3(predictors, targets):
    dense_model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(units=60, activation='relu', input_shape=(len(predictors),)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=60, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=len(targets))
    ])

def rolling_window(window_size, data, predictors, targets, model, eval_metrics, date, forecast_horizon):
    df = data
    df = df.dropna()

    original_df = df

    forecast_dataset = original_df.copy()
    forecast_targets = original_df[targets].shift(-forecast_horizon).add_prefix(f'{forecast_horizon}_day_forecast_')
    forecast_dataset = pd.concat([forecast_dataset, forecast_targets], axis=1).dropna()

    dense_model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(units=10, activation='relu', input_shape=(len(predictors),)),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=len(targets))
    ])

    model = dense_model1

    dataset = forecast_dataset

    row_start_index = int(dataset.index[0])
    row_end_index = int(dataset.index[-1])
    rows = row_end_index - row_start_index
    windows = rows // window_size

    y_actual_list = []
    y_predict_list = []
    date_index = []

    training_proportion = .7
    validation_proportion = .2
    testing_proportion = .1


    stride = window_size * testing_proportion

    MAX_EPOCHS = 50
    
    patience = 5

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=patience,
                                                mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=eval_metrics)

    for window_start_index in range(int(row_start_index), int(row_end_index - window_size), int(stride)):

        print(window_start_index, "of", int(row_end_index - window_size))

        window_end = window_start_index + window_size

        training_end = window_start_index + int(training_proportion * window_size)
        validation_end = training_end + int(validation_proportion * window_size)
        test_end = window_end

        training_data = dataset.drop(columns=date).loc[window_start_index:training_end]
        validation_data = dataset.drop(columns=date).loc[training_end:validation_end]
        testing_data = dataset.drop(columns=date).loc[validation_end:test_end]

        #standardize (independently)
        means_predictors = training_data[predictors].mean()
        stds_predictors = training_data[predictors].std()

        means_targets = training_data[targets].mean()
        stds_targets = training_data[targets].std()

        X_train = (training_data[predictors] - means_predictors[predictors]) / (stds_predictors[predictors] + 1e-6)
        y_train = (training_data[targets] - means_targets[targets]) / (stds_targets[targets] + 1e-6)

        X_val = (validation_data[predictors] - means_predictors[predictors]) / (stds_predictors[predictors] + 1e-6)
        y_val = (validation_data[targets] - means_targets[targets]) / (stds_targets[targets] + 1e-6)

        X_test = (testing_data[predictors] - means_predictors[predictors]) / (stds_predictors[predictors] + 1e-6)
        y_test = (testing_data[targets] - means_targets[targets]) / (stds_targets[targets] + 1e-6)


        #train

        model.fit(X_train, y_train, epochs=MAX_EPOCHS,
                    validation_data=(X_val,y_val),
                    callbacks=[early_stopping])


        #predict & unstandardize
        
        #predictions = model(test_data)

        y_pred_normalized = model.predict(X_test)
        y_pred = (y_pred_normalized * stds_targets[targets].values) + means_targets[targets].values

        y_actual = (y_test * stds_targets[targets].values) + means_targets[targets].values

        y_actual_list.extend(y_actual.values)
        y_predict_list.extend(y_pred)
        
        date_list = list(dataset[date].loc[validation_end:test_end].values.flatten())
        date_index.extend(date_list)


    #evaluate


    y_actual = np.array(y_actual_list)
    y_predict = np.array(y_predict_list)

    test_error = y_actual - y_predict

    test_mean_square_error = (test_error**2).mean()
    test_mean_absolute_error = abs(test_error).mean()
    test_mean_absolute_percent_error = abs(test_error/y_actual).mean() * 100

    print(test_mean_square_error)
    print(test_mean_absolute_error)
    print(test_mean_absolute_percent_error)

    results = pd.DataFrame({
        "Date": date_index,
        "y_actual": y_actual,
        "y_predict": y_predict
    })
    return results