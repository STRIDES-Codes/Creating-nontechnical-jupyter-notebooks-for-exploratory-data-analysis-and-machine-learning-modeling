import os
import sys
from collections import defaultdict

import target_class_lib as tc

project_directory = '/srv/codathon'

if os.path.join(project_directory, 'checkout') not in sys.path:
    sys.path.append(os.path.join(project_directory, 'checkout'))

links_dir = os.path.join(project_directory, 'data', 'all_gene_expression_files_in_target', 'links')
annotation_file = os.path.join(project_directory, 'data', 'gencode.v22.annotation.gtf')
sample_sheet_file = os.path.join(project_directory, 'data', 'gdc_sample_sheet.2020-07-02.tsv')
metadata_file = os.path.join(project_directory, 'data', 'metadata.cart.2020-07-02.json')

LABEL_COLUMN_NAME = 'label 1'

DEFAULT_LOG_FUNCTION = lambda *args: None

# TODO Define user inputs
## 


def clean_up_index_labels(df):
    # Replaces hyphens with underscores in labels
    df.index = df.index.str.replace('-', '_')
    return df


def get_data(log_fcn=DEFAULT_LOG_FUNCTION):
    # Loads the data downloaded from the GDC Data Portal
    df_samples, df_counts = tc.load_gdc_data(sample_sheet_file, metadata_file, links_dir, log_fcn=log_fcn)
    
    # Add a labels column based on the project id and sample type 
    # columns and show the unique values by decreasing frequency
    df_samples[LABEL_COLUMN_NAME] = df_samples['project id'] + ', ' + df_samples['sample type']

    clean_up_index_labels(df_samples)
    clean_up_index_labels(df_counts)

    df_fpkm, df_fpkm_uq = tc.get_fpkm(df_counts, annotation_file, df_samples, links_dir, log_fcn=log_fcn)
    return df_samples, df_counts, df_fpkm, df_fpkm_uq


def drop_multiperson_samples(samples, counts, fpkm, fpkm_uq, log_fcn=DEFAULT_LOG_FUNCTION):
    # Removes samples that correspond to multiple cases (i.e., people)
    df_samples, indices_to_keep, _ = tc.drop_multiperson_samples(samples)
    df_counts = counts.iloc[indices_to_keep,:]
    df_fpkm = fpkm.iloc[indices_to_keep,:]
    df_fpkm_uq = fpkm_uq.iloc[indices_to_keep,:]
    return df_samples, df_counts, df_fpkm, df_fpkm_uq


def filter_data_by_column_value(samples, counts, fpkm, fpkm_uq, column, allowed_values=None):
    if allowed_values is not None:
        indices_to_keep = samples[column].isin(allowed_values).values
        df_samples = samples[indices_to_keep]
        df_counts = counts[indices_to_keep]
        df_fpkm = fpkm[indices_to_keep]
        df_fpkm_uq = fpkm_uq[indices_to_keep]
        return df_samples, df_counts, df_fpkm, df_fpkm_uq
    else:
        return samples, counts, fpkm, fpkm_uq

    

def apply_cutoffs(samples, counts, fpkm, fpkm_uq, nstd_by_column=defaultdict(lambda: 2), log_fcn=DEFAULT_LOG_FUNCTION):
    df_samples, indices_to_keep = tc.remove_bad_samples(samples, nstd_by_column, log_fcn=log_fcn)
    df_counts = counts.iloc[indices_to_keep,:]
    df_fpkm = fpkm.iloc[indices_to_keep,:]
    df_fpkm_uq = fpkm_uq.iloc[indices_to_keep,:]
    return df_samples, df_counts, df_fpkm, df_fpkm_uq


def perform_eda(samples, log_fcn=DEFAULT_LOG_FUNCTION):
    # Performs exploratory data analysis on the sample labels
    tc.eda_labels(samples, log_fcn=log_fcn)


def calculate_tpm(counts, log_fcn=DEFAULT_LOG_FUNCTION):
    # Calculate the TPM using the counts and gene lengths
    df_tpm = tc.get_tpm(counts, annotation_file, log_fcn=log_fcn)
    df_tpm.shape
    return clean_up_index_labels(df_tpm)


def perform_pca(counts_dataframe, samples):
    labels_series = samples[LABEL_COLUMN_NAME]
    # Creates and plots PCA analyses using 
    # the variance-stabilizing-transformed data from DESeq2
    assert counts_dataframe.index.equals(labels_series.index)
    import sklearn.decomposition as sk_decomp
    pca = sk_decomp.PCA(n_components=10)
    pca_res = pca.fit_transform(counts_dataframe.iloc[:,:500])
    ax = tc.plot_unsupervised_analysis(pca_res, labels_series)
    ax.set_title('PCA - variance-stabilizing transformation')


def perform_tsne(counts_dataframe, samples):
    labels_series = samples[LABEL_COLUMN_NAME]
    # Creates and plots TSNE analyses using 
    # the variance-stabilizing-transformed data from DESeq2
    assert counts_dataframe.index.equals(labels_series.index)
    import sklearn.manifold as sk_manif
    tsne = sk_manif.TSNE(n_components=2)
    tsne_res = tsne.fit_transform(counts_dataframe.iloc[:,:500])  # TODO allow filtering?
    ax = tc.plot_unsupervised_analysis(tsne_res, labels_series)
    ax.set_title('tSNE - variance-stabilizing transformation')



if __name__ == '__main__':
    df_samples, df_counts, df_fpkm, df_fpkm_uq = get_data()

    
    # TODO allow selection of project id, tissues, types of cancer, categories
    
    # TODO display summary statistics

    # Plot histograms of the numerical columns of the samples/labels before 
    # and after cutoffs are applied, and print out a summary of what was removed
    perform_eda(df_samples)
    df_samples, df_counts, df_fpkm, df_fpkm_uq = apply_cutoffs(df_samples, df_counts, df_fpkm, df_fpkm_uq)
    perform_eda(df_samples)


    # TODO how will this be displayed/used? 
    # Print some random data for us to spot-check in the files 
    # themselves to manually ensure we have a handle on the data arrays
    tc.spot_check_data(df_samples, df_counts, df_fpkm, df_fpkm_uq)


    df_tpm = calculate_tpm(df_counts)

    # TODO Run the variance-stabilizing transformation using DESeq2
    # using this most-detailed set of labels


    perform_pca(df_tpm, df_samples)
    perform_tsne(df_tpm, df_samples)
