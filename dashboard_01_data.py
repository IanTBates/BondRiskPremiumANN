from pathlib import Path

import numpy as np
import pandas as pd

from shiny import App, render, ui
from shinywidgets import output_widget, render_widget 


class DataDashboard:
    def __init__(self):
        self.dataset1 = pd.read_csv("bloomberg_values.csv")
        self.dataset2 = pd.read_csv("transformed_econ_df.csv")
        self.dataset3_1 = pd.read_excel("Bonds list.xlsx", sheet_name="SelectedIDs(Values)")
        self.dataset3_2 = pd.read_excel("Bonds list.xlsx", sheet_name="PricesTable(Values)")



    def ui(self):
        page_contents = ui.page_fluid(
            ui.card(
                ui.card_header("Yields and stuff1"),
                ui.p("CARD 1 Body")
            ),
            ui.input_slider("slider", "Slider", min=0, max=100, value=[35, 65]),  
            ui.output_text_verbatim("value"),
            output_widget("plot"),

        )
        
        return(page_contents)
    

    def server(self, input, output, session):
        @render.text
        def value():
            return f"{input.slider()}"


        @render_widget  
        def plot():  
            scatterplot = px.histogram(
                data_frame=self.penguins,
                x="body_mass_g",
                nbins=input.n(),
            ).update_layout(
                title={"text": "Penguin Mass", "x": 0.5},
                yaxis_title="Count",
                xaxis_title="Body Mass (g)",
            )

            return scatterplot 



    def import_data(self):
        pass


