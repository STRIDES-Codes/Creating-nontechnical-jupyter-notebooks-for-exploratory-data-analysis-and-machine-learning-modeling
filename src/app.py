
import ipywidgets as widgets
from src.api import analysis as api
from src.api.target_class_lib import higher_is_better_by_column

PROJECT_ID_COLUMN = 'project id'
SAMPLE_TYPE_COLUMN = 'sample type'


class App:

    def __init__(self):
        self.container = widgets.Tab()
        self.add_data_tab()
        self.add_dim_reduction_tab()
        self.add_autoencoder_tab()
        self.log_tab = self.add_tab('Logs')
        self.log_context = self.log_tab.add_child()

    def log_fcn(self, *args):
        self.log_context.append_stdout(' '.join([str(arg) for arg in args]))

    def add_dim_reduction_tab(self):
        self.dim_reduction_tab = self.add_tab('PCA/tSNE')

        pca_btn = widgets.Button(description='Perform PCA')
        self.dim_reduction_tab.add_child(pca_btn)
        self.pca_ctx = self.dim_reduction_tab.add_child()

        def pca(_):
            self.pca_ctx.clear_output()
            with self.pca_ctx:
                api.perform_pca(self.df_tpm, self.df_samples)

        pca_btn.on_click(pca)
        
        tsne_btn = widgets.Button(description='Perform tSNE')

        def tsne(_):
            self.tsne_ctx.clear_output()
            with self.tsne_ctx:
                api.perform_tsne(self.df_tpm, self.df_samples)

        self.dim_reduction_tab.add_child(tsne_btn)
        self.tsne_ctx = self.dim_reduction_tab.add_child()
        tsne_btn.on_click(tsne)

    
    def add_autoencoder_tab(self):
        self.autoencoder_tab = self.add_tab('Autoencoder')

        pca_btn = widgets.Button(description='Run Autoencoder with PCA', width=600)
        self.autoencoder_tab.add_child(pca_btn)
        self.autoencoder_pca_ctx = self.dim_reduction_tab.add_child()

        def pca(_):
            self.autoencoder_pca_ctx.clear_output()
            with self.pca_ctx:
                api.run_autoencoder(self.df_tpm, self.df_samples, dim_reduction_method='PCA', n_components=10)

        pca_btn.on_click(pca)
        
        tsne_btn = widgets.Button(description='Run Autoencoder with tSNE')

        def tsne(_):
            self.autoencoder_tsne_ctx.clear_output()
            with self.autoencoder_tsne_ctx:
                api.run_autoencoder(self.df_tpm, self.df_samples, dim_reduction_method='PCA', n_components=10)

        self.autoencoder_tab.add_child(tsne_btn)
        self.autoencoder_tsne_ctx = self.autoencoder_tab.add_child()
        tsne_btn.on_click(tsne)

    def add_data_tab(self):
        self.data_tab = self.add_tab('Data')

        self.data_selection_btn = widgets.Button(description='Load Data', icon='refresh')
        self.data_selection_btn.on_click(self.filter_data_on_request)
        self.data_tab.add_child(self.data_selection_btn)

        # Initialize data
        self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq = api.get_data()
        self.df_tpm = api.calculate_tpm(self.df_counts)

        self.add_data_selectors()

        # Add a handle for plots of selected data
        self.eda_ctx = self.data_tab.add_child()

    def add_data_selectors(self):
        # Add project and sample type selctors
        self.project_selectors = self.create_categorical_value_selectors(PROJECT_ID_COLUMN)
        self.sample_type_selectors = self.create_categorical_value_selectors(SAMPLE_TYPE_COLUMN)

        project_selector_box = widgets.VBox(
            children=[widgets.HTML('<h2>Select Projects</h2>')] + list(self.project_selectors.values())
        )
        sample_type_selector_box = widgets.VBox(
            children=[widgets.HTML('<h2>Select Sample Types</h2>')] + list(self.sample_type_selectors.values()),
            min_width=1200
        )
        nstd_box = self.add_sliders_to_filter_data_by_nstd()
        
        selector_group = widgets.HBox(children=[project_selector_box, sample_type_selector_box])
        self.data_tab.add_child(selector_group)
        self.data_tab.add_child(nstd_box)

    def add_sliders_to_filter_data_by_nstd(self):
        self.nstd_sliders = {}
        box = widgets.VBox(children=[widgets.HTML('<h2>Remove Outliers Based on Standard Deviation</h2>')])

        for column in higher_is_better_by_column.keys():
            box.children += (widgets.Label(value=column),)

            slider = widgets.FloatSlider(
                value=2.0,
                min=0,
                max=10.0,
                step=0.1,
                description='',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
            self.nstd_sliders[column] = slider
            box.children += (slider,)
        
        return box

    def filter_data_on_request(self, _):
        self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq = api.get_data()

        self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq = api.filter_data_by_column_value(
            self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq, PROJECT_ID_COLUMN, 
            allowed_values=[key for key, widget in self.project_selectors.items() if widget.value]
        )

        self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq = api.filter_data_by_column_value(
            self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq, SAMPLE_TYPE_COLUMN, 
            allowed_values=[key for key, widget in self.sample_type_selectors.items() if widget.value]
        )

        self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq = api.apply_cutoffs(
            self.df_samples, self.df_counts, self.df_fpkm, self.df_fpkm_uq, 
            {column: widget.value for column, widget in self.nstd_sliders.items()}
        )

        self.eda_ctx.clear_output(wait=True)
        with self.eda_ctx:
            api.perform_eda(self.df_samples, log_fcn=self.log_fcn)
        
        self.df_tpm = api.calculate_tpm(self.df_counts)

    def add_tab(self, tab_title):
        tab = Tab()
        self.container.children += (tab.container,)
        tab_idx = len(self.container.children)
        self.container.set_title(tab_idx - 1, tab_title)
        return tab

    def create_categorical_value_selectors(self, column_name):
        allowed_values = self.df_samples[column_name].unique()
        selectors = {}
        for value in allowed_values:
            selector = widgets.Checkbox(
                value=True,
                description=value,
                disabled=False,
                indent=False,
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

