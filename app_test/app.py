from shiny import App, ui, render, reactive, Inputs, Outputs, Session
import pandas as pd
import plotly.express as px
import os

BASE_DIR = os.path.dirname(__file__)
DATA_FILES = [
    f for f in os.listdir(BASE_DIR)
    if f.endswith(('.csv', '.xlsx', '.xls')) and os.path.isfile(os.path.join(BASE_DIR, f))
]

app_ui = ui.page_fluid(
    ui.h2("Data Explorer (CSV / Excel)"),

    ui.input_select("file_select", "Select File", choices=DATA_FILES),
    ui.output_ui("sheet_selector"),

    ui.hr(),
    ui.output_ui("column_selectors"),

    ui.hr(),
    ui.h4("Data Preview"),
    ui.output_table("data_table"),

    ui.hr(),
    ui.h4("Time Series Plot"),
    ui.output_ui("x_selector"),
    ui.output_ui("y_selector"),
    ui.output_ui("ts_plot"),  # <-- Just generic ui.output_ui here
)

def server(input: Inputs, output: Outputs, session: Session):

    @reactive.Calc
    def file_path():
        file = input.file_select()
        return os.path.join(BASE_DIR, file) if file else None

    @reactive.Calc
    def is_excel():
        path = file_path()
        return path is not None and path.endswith((".xlsx", ".xls"))

    @reactive.Calc
    def sheet_names():
        if is_excel():
            try:
                xls = pd.ExcelFile(file_path())
                return xls.sheet_names
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                return []
        return []

    @output
    @render.ui
    def sheet_selector():
        if is_excel():
            return ui.input_select("sheet_select", "Select Sheet", choices=sheet_names())
        return None

    @reactive.Calc
    def df():
        path = file_path()
        if not path:
            return pd.DataFrame()
        try:
            if path.endswith(".csv"):
                return pd.read_csv(path)
            elif path.endswith((".xlsx", ".xls")):
                sheet = input.sheet_select() if "sheet_select" in input else None
                if sheet:
                    return pd.read_excel(path, sheet_name=sheet)
        except Exception as e:
            print(f"Error reading data: {e}")
            return pd.DataFrame()
        return pd.DataFrame()

    @output
    @render.ui
    def column_selectors():
        columns = df().columns.tolist()
        if not columns:
            return None
        return ui.div(
            ui.input_selectize("predictors", "Select Predictors", choices=columns, multiple=True),
            ui.input_select("target", "Select Target", choices=columns)
        )

    @output
    @render.ui
    def x_selector():
        columns = df().columns.tolist()
        if not columns:
            return None
        return ui.input_select("x_col", "X Axis", choices=columns)

    @output
    @render.ui
    def y_selector():
        columns = df().columns.tolist()
        if not columns:
            return None
        return ui.input_selectize("y_cols", "Y Axis Column(s)", choices=columns, multiple=True)

    @output
    @render.table
    def data_table():
        return df().head(100)

    @output
    @render.ui
    def ts_plot():
        d = df()
        x = input.x_col()
        y = input.y_cols()

        if not x or not y:
            return ui.HTML("<p>Please select X and Y columns to plot.</p>")

        # Defensive conversion of y to list
        if isinstance(y, tuple):
            y = list(y)
        elif isinstance(y, str):
            y = [y]

        try:
            fig = px.line(d, x=x, y=y, title="Time Series Plot")
            return ui.plotly(fig)  # <-- wrap plotly figure here
        except Exception as e:
            print(f"Error in plot: {e}")
            return ui.HTML(f"<p>Error generating plot: {e}</p>")

app = App(app_ui, server)
