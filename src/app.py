
import ipywidgets as widgets
from ipywidgets.widgets import widget
from src.api import analysis as api

class App:

    def __init__(self):
        self.container = widgets.Tab()
        self.data_tab = self.add_tab('Data')
        self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq = api.get_data()
        self.project_selectors = self.create_categorical_value_selectors(self.data_tab, 'project id')
        self.sample_type_selectors = self.create_categorical_value_selectors(self.data_tab, 'sample type')

        project_selector_box = widgets.VBox(
            children=[widgets.Label(value='Select Projects')] + list(self.project_selectors.values())
        )
        sample_type_selector_box = widgets.VBox(
            children=[widgets.Label(value='Select Sample Types')] + list(self.sample_type_selectors.values())
        )
        selector_group = widgets.HBox(children=[project_selector_box, sample_type_selector_box])
        self.data_tab.add_child(selector_group)

    def add_tab(self, tab_title):
        tab = Tab()
        self.container.children += (tab.container,)
        tab_idx = len(self.container.children)
        self.container.set_title(tab_idx - 1, tab_title)
        return tab

    def create_categorical_value_selectors(self, tab, column_name):
        allowed_values = self.df_samples[column_name].unique()
        selectors = {}
        for value in allowed_values:
            selector = widgets.Checkbox(
                value=True,
                description=value,
                disabled=False,
                indent=False
            )
            selectors[value] = selector
        return selectors



class Tab:

    def __init__(self):
        self.container = widgets.VBox()

    def add_child(self, child = None):
        if not child:
            child = widgets.Output()
        self.container.children += (child,)
        return child

