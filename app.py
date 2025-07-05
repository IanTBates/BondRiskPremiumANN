
import numpy
import pandas

import shiny

from shiny import App, render, ui

from shiny import App, ui

from yielddashboard import yielddashboard as yield_db

yield_db = yield_db()

app_ui = ui.page_navbar(  
    ui.nav_panel("Yield Analyis", yield_db.page()),  
    ui.nav_panel("Data Analysis", "Page B content"),  
    ui.nav_panel("Artificial Neural Network", "Page C content"),  
    title="Bond Risk Premium Analysis App",  
    id="page",  
)  




def server(input, output, session):
    pass


app = App(app_ui, server)


if __name__ == "__main__":
    pass
    

