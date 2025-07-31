
import numpy
import pandas

import shiny

from shiny import App, render, ui

from shiny import App, ui

from dashboard_01_data import DataDashboard as data_db
from dashboard_02_yield import YieldDashboard as yield_db
from dashboard_03_ann import ANNDashboard as ann_db

yield_db = yield_db()
data_db = data_db()
ann_db = ann_db()

app_ui = ui.page_navbar(  
    ui.nav_panel("Data Analysis", data_db.ui()),  
    ui.nav_panel("Yield Analyis", yield_db.ui()), 
    ui.nav_panel("Artificial Neural Network", ann_db.ui()),  
    title="Fixed income Machine Learning App",  
    id="page",  
)  

def server(input, output, session):
    data_db.server(input,output,session)
    yield_db.server(input,output,session)
    ann_db.server(input,output,session)
app = App(app_ui, server)
    

