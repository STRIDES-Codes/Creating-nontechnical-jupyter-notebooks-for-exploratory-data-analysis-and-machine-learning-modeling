import os
import sys
from collections import defaultdict

from numpy.lib.function_base import _DIMENSION_NAME

from src.api import target_class_lib as tc

project_directory = '/srv/codathon'

if os.path.join(project_directory, 'checkout') not in sys.path:
    sys.path.append(os.path.join(project_directory, 'checkout'))

links_dir = os.path.join(project_directory, 'data', 'all_gene_expression_files_in_target', 'links')
annotation_file = os.path.join(project_directory, 'data', 'gencode.v22.annotation.gtf')
sample_sheet_file = os.path.join(project_directory, 'data', 'gdc_sample_sheet.2020-07-02.tsv')
metadata_file = os.path.join(project_directory, 'data', 'metadata.cart.2020-07-02.json')

LABEL_COLUMN_NAME = 'label 1'

AUTO_ENCODER_MODEL_PATH = 'src/models/autoencoder_model.h5'

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


def perform_pca(counts_dataframe, samples, n_components=10):
    labels_series = samples[LABEL_COLUMN_NAME]
    # Creates and plots PCA analyses
    import sklearn.decomposition as sk_decomp
    pca = sk_decomp.PCA(n_components=n_components)
    pca_res = pca.fit_transform(counts_dataframe.iloc[:,:500])
    ax = tc.plot_unsupervised_analysis(pca_res, labels_series)
    ax.set_title('PCA')


def perform_tsne(counts_dataframe, samples, n_components=2):
    labels_series = samples[LABEL_COLUMN_NAME]
    # Creates and plots tSNE analyses
    import sklearn.manifold as sk_manif
    tsne = sk_manif.TSNE(n_components=n_components)
    tsne_res = tsne.fit_transform(counts_dataframe.iloc[:,:500])  # TODO allow filtering?
    ax = tc.plot_unsupervised_analysis(tsne_res, labels_series)
    ax.set_title('tSNE')


def generate_autoencoder_model(counts_dataframe, samples, encoding_dim=600, decoding_dim=2000, epochs=5, batch_size=100):
    from keras.layers import Input, Dense
    from keras.models import Model
    from sklearn.model_selection import train_test_split
    from numpy.random import seed
    from sklearn import preprocessing
    import pandas as pd
    import sklearn.decomposition as sk_decomp
    import sklearn.manifold as sk_manif

    X1 = counts_dataframe.sort_index()
    y1 = samples[LABEL_COLUMN_NAME].sort_index()
    y1_num=y1.copy()

    #encode the cancer type and store the coding in y1_dic
    y1_v=y1_num.value_counts().index.tolist()
    y1_dic = {k: v for v, k in enumerate(y1_v)}

    #covert cancer types based on the coding 
    for k, v in y1_dic.items():
        y1_num.replace(k, v, inplace = True)

    #normalize each sample first and standardize each feature independently (zero mean and unit variance)
    #this is further used for model training
    X1_std = preprocessing.scale(X1)
    X1_std_norm = preprocessing.normalize(X1_std)
    X1_std_norm = pd.DataFrame(data=X1_std_norm, index=X1.index, columns=X1.columns)

    #standardize each feature independently (zero mean and unit variance) first and normalize each sample
    X1_norm = preprocessing.normalize(X1)
    X1_norm_std = preprocessing.scale(X1_norm)
    X1_norm_std = pd.DataFrame(data=X1_norm_std, index=X1.index, columns=X1.columns)

    #set up autoencoder
    #split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X1_std_norm, y1_num, train_size = 0.7, random_state = seed(42))
    ncol = X1.shape[1]
    input_dim = Input(shape = (ncol, ))
    #define the dimension of encoder
    #define the encoder layer
    encoded1 = Dense(decoding_dim, activation = 'relu')(input_dim)
    encoded2 = Dense(encoding_dim, activation = 'relu')(encoded1)
    #define the decoder layer
    decoded1 = Dense(decoding_dim, activation = 'sigmoid')(encoded2)
    decoded2 = Dense(ncol, activation = 'sigmoid')(decoded1)
    #combine encoder and decoder into an autoencoder model
    autoencoder = Model(input_dim, decoded2)
    #configure and train the autoencoder
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
    autoencoder.summary()
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))

    # saving whole model
    autoencoder.save(AUTO_ENCODER_MODEL_PATH)
    return autoencoder

    
def run_autoencoder(counts_dataframe, samples, dim_reduction_method='PCA', n_components=10):
    # loading whole model
    from keras.models import load_model
    if not os.path.exists(AUTO_ENCODER_MODEL_PATH):
        encoder = generate_autoencoder_model(counts_dataframe, samples)
    else:
        encoder = load_model(AUTO_ENCODER_MODEL_PATH)
    
    import pandas as pd
    encoded_out = encoder.predict(counts_dataframe)
    if dim_reduction_method == 'PCA':
        perform_pca(pd.DataFrame(encoded_out), samples, n_components=n_components)
    elif dim_reduction_method == 'tSNE':
        perform_tsne(pd.DataFrame(encoded_out), samples, n_components=n_components)

