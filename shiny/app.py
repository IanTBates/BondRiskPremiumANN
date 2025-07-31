
#import numpy as np
import pandas as pd

import plotly.express as px

import shiny

from shiny import App, render, ui, reactive
from shinywidgets import output_widget, render_widget, render_plotly

#import time_series_regression as tsr

import os

app_dir = os.path.dirname(os.path.abspath(__file__))
data_path1 = os.path.join(app_dir, 'bloomberg_values.csv')
data_path2 = os.path.join(app_dir, 'transformed_econ_df.csv')
data_path3_1 = os.path.join(app_dir, 'Bonds list.xlsx')
data_path3_2 = os.path.join(app_dir, 'Bonds list.xlsx')
model_results_path = os.path.join(app_dir, 'model_results.csv')

dataset1 = pd.read_csv(data_path1)
dataset2 = pd.read_csv(data_path2)
dataset3_1 = pd.read_excel(data_path3_1, sheet_name="SelectedIDs(Values)")
dataset3_2 = pd.read_excel(data_path3_2, sheet_name="PricesTable(Values)")
model_results = pd.read_csv(model_results_path)
#rsconnect add --account <ACCOUNT> --name <NAME> --token <TOKEN> --secret <SECRET>
#rsconnect deploy shiny C:\Users\Iantb\Documents\GitHub\BondRiskPremiumANN\shiny --name iantbates --title my-app

datasets = {
    "Bloomberg": dataset1,
    "Econ": dataset2,
    "Bond Features": dataset3_1,
    "Bond Prices": dataset3_2
}

data = reactive.Value()
date_column = reactive.Value()

model_choices = {
    "Linear Model": 1, 
    "Dense Model 1": 2, 
    "Dense Model 2": 3, 
    "Dense Model 3": 4, 
}

eval_metrics = {
    "MAE": 1,
    "MAPE": 2,
    "MSE": 3
}

# UI
app_ui = ui.page_navbar(
    ui.nav_panel("Data Analysis",
        ui.page_sidebar(
            ui.sidebar(
                ui.card(
                    ui.output_ui("date"),
                    #ui.input_text("date_format", "Date Format (optional, like %d-%m-%Y)", value=""),
                    #ui.input_action_button("convert_datetime", "Convert"),
                    #ui.output_text("convert_on_button_click")
                )
            ),
            ui.card(
                ui.input_select("select_data", "Select Dataset", choices=list(datasets.keys())),
                ui.output_data_frame("table"),
                height=900
            )
           
        )
    ),
    ui.nav_panel("Yield Analysis",
        ui.page_sidebar(
            ui.sidebar(
                ui.card(
                    ui.output_ui("y_plot"),
                    ui.output_ui("x_plot"),
                    #ui.input_action_button("go_plot", "Plot")
                )
            ),
            ui.card(
            output_widget("exploratory_plot")
            )
        )
    ),
    ui.nav_panel("Artificial Neural Network",
        ui.page_sidebar(
            ui.sidebar(
                ui.input_numeric("input_lag_list","Lags",value=1),
                ui.input_numeric("input_forecasting_horizons","Forecast Horizons", value = 5),
                ui.input_numeric("input_window_size","Window Size", value = 300),
                ui.input_checkbox_group("checked_eval_metrics","Eval Metrics", choices=list(eval_metrics.keys())),
                ui.input_select("selected_model", "Model", choices= model_choices)
                ),
            ui.card(
                ui.layout_columns(
                    ui.output_ui("targets"),
                    ui.output_ui("predictors")
                ),
                height=300
            ),
            ui.card(
            output_widget("results_plot")
            )
        )
    ),
    title="Fixed Income Machine Learning App",
    id="page"
) 

def server(input, output, session):
    @reactive.Calc
    def dataset():
        key = input.select_data()
        if key in datasets:
            df = datasets[key]
            data.set(df)
            return df
        return pd.DataFrame()

    @output
    @render.data_frame
    def table():
        return render.DataGrid(dataset())

    @output
    @render.ui
    def date():
        df = data.get()
        if df is None or df.empty:
            return None
        return ui.input_select("date_column", "Choose Date Column", choices=df.columns.tolist(),multiple=True)

    @output
    @render.ui
    def targets():
        df = data.get()
        if df is None or df.empty:
            return None
        return ui.input_selectize("target_columns", "Choose Targets", choices=df.columns.tolist(),multiple=True, width= '30%')

    @output
    @render.ui
    def predictors():
        df = data.get()
        if df is None or df.empty:
            return None
        return ui.input_selectize("predictor_columns", "Choose Predictors", choices=df.columns.tolist(),multiple=True, width='30%')
    
    @output
    @render.ui
    def y_plot():
        df = data.get()
        if df is None or df.empty:
            return None
        return ui.input_select("y_values_plot", "Choose Y", choices=df.columns.tolist())

    @output
    @render.ui
    def x_plot():
        df = data.get()
        if df is None or df.empty:
            return None
        return ui.input_select("x_values_plot", "Choose X", choices=df.columns.tolist())

    @output
    @render_widget
    def exploratory_plot():
        df = data.get()
        # Inputs must be lists (multi-select)
        x_col = "Date"#input.x_values_plot()
        y_col = "EUDR1T Curncy"#input.y_values_plot()

        x_col = input.x_values_plot.get()
        y_col = input.y_values_plot.get()

        print(x_col)
        print(y_col)
        fig = px.line(
            df,
            x=x_col,
            y=y_col#,
            #title=f"{y_col} vs {x_col}"
        ).update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        return fig
    
    @output
    @render_widget
    def results_plot():
        df = model_results

        x_col = "Date"
        y_cols = ["y_actual 1", "y_predict 1"]  # Replace with your actual column names
        
        # Make sure columns exist
        if not all(col in df.columns for col in [x_col] + y_cols):
            return None

        # Reshape for multi-line plot
        df_long = df.melt(id_vars=x_col, value_vars=y_cols,
                        var_name="Type", value_name="Value")

        fig = px.line(
            df_long,
            x=x_col,
            y="Value",
            color="Type",
            title="Actual vs Predicted over Time"
        ).update_layout(
            xaxis_title=x_col,
            yaxis_title="Value"
        )

        return fig
app = App(app_ui, server)
    

