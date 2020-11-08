
import ipywidgets as widgets

class App:

    def __init__(self):
        self.container = widgets.Tab()

    def add_tab(self, tab_title):
        tab = Tab()
        self.container.children += (tab.container,)
        tab_idx = len(self.container.children)
        self.container.set_title(tab_idx - 1, tab_title)
        return tab

class Tab:

    def __init__(self):
        self.container = widgets.VBox()

    def add_child(self, child = None):
        if not child:
            child = widgets.Output()
        self.container.children += (child,)
        return child

