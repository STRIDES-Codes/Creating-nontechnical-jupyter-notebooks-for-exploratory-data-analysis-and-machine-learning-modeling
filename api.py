import os, sys


import os

import target_class_lib as tc


project_directory = '/srv/codathon'

if os.path.join(project_directory, 'checkout') not in sys.path:
    sys.path.append(os.path.join(project_directory, 'checkout'))

links_dir = os.path.join(project_directory, 'data', 'all_gene_expression_files_in_target', 'links')
annotation_file = os.path.join(project_directory, 'data', 'gencode.v22.annotation.gtf')
sample_sheet_file = os.path.join(project_directory, 'data', 'gdc_sample_sheet.2020-07-02.tsv')
metadata_file = os.path.join(project_directory, 'data', 'metadata.cart.2020-07-02.json')

LABEL_COLUMN_NAME = 'label 1'

# TODO Define user inputs
## 

def get_data():
    # Loads the data downloaded from the GDC Data Portal
    df_samples, df_counts = tc.load_gdc_data(sample_sheet_file, metadata_file, links_dir)
    
    # Add a labels column based on the project id and sample type 
    # columns and show the unique values by decreasing frequency
    df_samples[LABEL_COLUMN_NAME] = df_samples['project id'] + ', ' + df_samples['sample type']

    df_fpkm, df_fpkm_uq = tc.get_fpkm(df_counts, annotation_file, df_samples, links_dir)
    return df_samples, df_counts, df_fpkm, df_fpkm_uq


def drop_multiperson_samples(samples, counts, fpkm, pkm_uq):
    # Removes samples that correspond to multiple cases (i.e., people)
    df_samples, indexes_to_keep, _ = tc.drop_multiperson_samples(samples)
    df_counts = counts.iloc[indexes_to_keep,:]
    df_fpkm = fpkm.iloc[indexes_to_keep,:]
    df_fpkm_uq = pkm_uq.iloc[indexes_to_keep,:]
    return df_samples, df_counts, df_fpkm, df_fpkm_uq


def apply_cutoffs(samples, counts, fpkm, pkm_uq):
    df_samples, indexes_to_keep = tc.remove_bad_samples(samples)
    df_counts = counts.iloc[indexes_to_keep,:]
    df_fpkm = fpkm.iloc[indexes_to_keep,:]
    df_fpkm_uq = pkm_uq.iloc[indexes_to_keep,:]
    return df_samples, df_counts, df_fpkm, df_fpkm_uq


def perform_eda(samples):
    # Performs exploratory data analysis on the sample labels
    tc.eda_labels(samples)


def clean_up_index_labels(df):
    # Replaces hyphens with underscores in labels
    df.index = df.index.str.replace('-', '_')
    return df


def calculate_tpm(counts):
    # Calculate the TPM using the counts and gene lengths
    df_tpm = tc.get_tpm(df_counts, annotation_file)
    df_tpm.shape
    return clean_up_index_labels(df_tpm)


def perform_pca(counts_dataframe, labels_series):
    # Creates and plots PCA analyses using 
    # the variance-stabilizing-transformed data from DESeq2
    assert counts_dataframe.index.equals(labels_series.index)
    import sklearn.decomposition as sk_decomp
    pca = sk_decomp.PCA(n_components=10)
    pca_res = pca.fit_transform(counts_dataframe)
    print('Top {} PCA explained variance ratios: {}'.format(10, pca.explained_variance_ratio_))
    ax = tc.plot_unsupervised_analysis(pca_res, labels_series)
    ax.set_title('PCA - variance-stabilizing transformation')


def perform_tsne(counts_dataframe, labels_series):
    # Creates and plots TSNE analyses using 
    # the variance-stabilizing-transformed data from DESeq2
    assert counts_dataframe.index.equals(labels_series.index)
    import sklearn.manifold as sk_manif
    tsne = sk_manif.TSNE(n_components=2)
    tsne_res = tsne.fit_transform(counts_dataframe)  # TODO allow filtering?
    ax = tc.plot_unsupervised_analysis(tsne_res, labels_series)
    ax.set_title('tSNE - variance-stabilizing transformation')



if __name__ == '__main__':
    df_samples, df_counts, df_fpkm, df_fpkm_uq = get_data()

    # TODO display summary statistics


    # Plot histograms of the numerical columns of the samples/labels before 
    # and after cutoffs are applied, and print out a summary of what was removed
    perform_eda(df_samples)
    apply_cutoffs(df_samples, df_counts, df_fpkm, df_fpkm_uq)
    perform_eda(df_samples)


    # TODO how will this be displayed/used? 
    # Print some random data for us to spot-check in the files 
    # themselves to manually ensure we have a handle on the data arrays
    tc.spot_check_data(df_samples, df_counts, df_fpkm, df_fpkm_uq)


    df_tpm = calculate_tpm(df_counts)


    clean_up_index_labels(df_samples)
    clean_up_index_labels(df_counts)

    # TODO Run the variance-stabilizing transformation using DESeq2
    # using this most-detailed set of labels

    
    perform_pca(df_tpm, df_samples[LABEL_COLUMN_NAME])
    perform_tsne(df_tpm, df_samples[LABEL_COLUMN_NAME])