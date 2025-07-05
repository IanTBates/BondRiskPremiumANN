
from shiny import App, render, ui

class yielddashboard:
    def yielddashboard(self):
        x = 20

    def page(self):
        page_contents = ui.page_fluid(
            ui.card(
                ui.card_header("Yields and stuff1"),
                ui.p("CARD 1 Body")
            )

        )
        
        return(page_contents)