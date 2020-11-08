# Prepare the annotation dataframe df_gencode_genes, particularly calculating the exon length of each gene (corresponding to its non-overlapping exons) and adding this as a column to the df_gencode_genes dataframe
# This takes about 10 minutes if the pickle file doesn't already exist
def calculate_exon_lengths(gencode_gtf_file, log_fcn=print):

    # Import relevant libraries
    import os

    import numpy as np
    import pandas as pd
    tci = get_tci_library()

    # Set the number of steps to output so we can evaluate progress
    nsteps = 100

    # Identify the data directory as the directory that the annotation file is in
    os_sep = os.sep
    data_dir = os_sep.join(gencode_gtf_file.split(sep=os_sep)[:-1])

    # If the file containing the annotation dataframe does not already exist...
    if not os.path.exists(os.path.join(data_dir,'annotation_dataframe.pkl')):

        # Read in the the GTF file from the Gencode website
        df_gencode = pd.read_csv(gencode_gtf_file, sep='\t', skiprows=5, header=None)
        df_gencode_genes = df_gencode[df_gencode[2]=='gene'].reset_index(drop=True)
        df_gencode_exons = df_gencode[df_gencode[2]=='exon'].reset_index(drop=True)

        # Format the df_gencode_genes dataframe for consistency
        df_gencode_genes['id'] = df_gencode_genes.apply(lambda x: x[8].split()[1].split('\"')[1], axis=1)
        df_gencode_genes['type'] = df_gencode_genes.apply(lambda x: x[8].split()[3].split('\"')[1], axis=1)
        df_gencode_genes['name'] = df_gencode_genes.apply(lambda x: x[8].split()[7].split('\"')[1], axis=1)
        df_gencode_genes = df_gencode_genes.rename({3: 'start', 4: 'end', 6: 'strand', 0: 'seqname'}, axis='columns')
        df_gencode_genes = df_gencode_genes.set_index('id')
        df_gencode_genes = df_gencode_genes.sort_index()

        # Format the df_gencode_exons dataframe for consistency
        # Takes about a minute
        df_gencode_exons['id'] = df_gencode_exons.apply(lambda x: x[8].split()[1].split('\"')[1], axis=1)
        df_gencode_exons['type'] = df_gencode_exons.apply(lambda x: x[8].split()[3].split('\"')[1], axis=1)
        df_gencode_exons['name'] = df_gencode_exons.apply(lambda x: x[8].split()[7].split('\"')[1], axis=1)
        df_gencode_exons = df_gencode_exons.rename({3: 'start', 4: 'end', 6: 'strand', 0: 'seqname'}, axis='columns')
        df_gencode_exons = df_gencode_exons.set_index('id')
        df_gencode_exons = df_gencode_exons.sort_index()

        # Set the step size in units of the size of the df_gencode_exons dataframe
        unit_len = int(len(df_gencode_exons) / nsteps)

        # Initialize some values
        istep = 0 # the step that we're on
        exon_lengths = [] # the array holding the final exon gene lengths (non-overlapping union of exon base pairs)
        prev_idx = '' # set the previous index to null

        # For every index in the ordered-by-index exons dataframe...
        for iidx, idx in enumerate(df_gencode_exons.index):

            # Get the current row of data in the dataframe
            curr_row = df_gencode_exons.iloc[iidx,:]

            # Output progress if the time is right
            if (iidx%unit_len) == 0:
                log_fcn('{}/{} complete...'.format(istep,nsteps))
                istep = istep + 1

            # If the current index is not equal to the previous index...
            if idx != prev_idx:

                # If the previous index is not null (i.e., if this isn't the very first loop iteration and therefore base_pairs has been initialized below), calculate and store the number of unique base pairs for the current unique idx
                if prev_idx != '':
                    exon_lengths.append(len(set(np.concatenate(base_pairs))))

                # Initialize the base_pairs holder (which will ultimately be a list of lists of base pairs)
                base_pairs = []

            # Always append the current set of base pairs corresponding to curr_row to the base_pairs list
            base_pairs.append(np.arange(curr_row['start'], curr_row['end']+1))

            # Set the previous index to the current index
            prev_idx = idx

        # Calculate and store the number of unique base pairs for the final unique idx
        exon_lengths.append(len(set(np.concatenate(base_pairs))))

        # Add a column of exon gene length to the genes dataframe
        df_gencode_genes['exon_length'] = exon_lengths

        tci.make_pickle(df_gencode_genes, data_dir, 'annotation_dataframe.pkl')

    # Otherwise, read it in
    else:
        df_gencode_genes = tci.load_pickle(data_dir, 'annotation_dataframe.pkl', log_fcn=log_fcn)

    # Return the main reference dataframe
    return(df_gencode_genes)


# Import the time cell interaction library
def get_tci_library():
    import sys
    gmb_dir = '/data/BIDS-HPC/private/projects/gmb/checkout'
    if gmb_dir not in sys.path:
        sys.path.append(gmb_dir)
    from src.api import time_cell_interaction_lib as tci  # we need this to get the pickle functions e.g.
    return(tci)


# Get a list of series containing the actual counts of the samples
def get_counts_old(links_dir):
    # Sample call: srs_counts = tc.get_counts('/data/BIDS-HPC/private/projects/dmi2/data/all_gene_expression_files_in_target/links')

    # Import relevant libraries
    import os

    import pandas as pd

    # Define the sample HT-Seq datafiles
    file_counts = 'fffee315-9aa3-44d2-8c89-78a2c1d107e7.htseq_counts.txt'

    # Get a sample list of counts series
    srs_counts = [pd.read_csv(os.path.join(links_dir, file_counts), sep='\t', skipfooter=5, names=['id','intensity'])]

    # Format the sample series for consistency
    for isr, sr in enumerate(srs_counts):
        sr = sr.set_index('id')
        srs_counts[isr] = sr.sort_index().iloc[:,0]

    # Return a list of series of counts
    return(srs_counts)


# Given the counts for a sample, calculate the FPKM and FPKM-UQ values using the protein-coding genes for normalization
def calculate_fpkm(df_gencode_genes, sr_counts):

    # Import relevant library
    import numpy as np

    # Get the number of reads and gene lengths for the entire set of genes
    exon_lengths = df_gencode_genes['exon_length']

    # Get the number of reads and gene lengths for just the protein-coding genes
    pc_loc = df_gencode_genes['type'] == 'protein_coding'
    pc_counts = sr_counts[pc_loc]
    #pc_lengths = exon_lengths[pc_loc]

    # Calculate the normalizations for the FPKM and FPKM-UQ values
    pc_frag_count = pc_counts.sum()
    upper_quantile = np.percentile(pc_counts, 75) # equals pc_counts.sort_values()[int(pc_loc.sum()*.75)]

    # Calculate the normalized counts via https://github.com/NCI-GDC/htseq-tool/blob/master/htseq_tools/tools/fpkm.py
    tmp = sr_counts / exon_lengths * 1e9
    fpkm = tmp / pc_frag_count
    fpkm_uq = tmp / upper_quantile

    # Return the normalized counts
    return(fpkm, fpkm_uq)


# Run a few checks on some known data
def run_checks(df_gencode_genes, calculated_counts, fpkm, fpkm_uq, log_fcn=print):

    # Sample call: tc.run_checks(df_gencode_genes, srs_counts[0], fpkm, fpkm_uq)

    # Import relevant libraries
    import os

    import pandas as pd

    # Calculate the constants
    gdc_tsv_file = '/data/BIDS-HPC/private/projects/dmi2/data/gencode.gene.info.v22.tsv'
    file_fpkm = 'fffee315-9aa3-44d2-8c89-78a2c1d107e7.FPKM.txt'
    file_fpkm_uq = 'fffee315-9aa3-44d2-8c89-78a2c1d107e7.FPKM-UQ.txt'
    links_dir = '/data/BIDS-HPC/private/projects/dmi2/data/all_gene_expression_files_in_target/links'

    # Read in and process the final TSV file that GDC uses and contains exon lengths whose results we want to check against (our calculated values in df_gencode_genes)
    df_gdc = pd.read_csv(gdc_tsv_file, sep='\t')
    df_gdc = df_gdc.rename({'gene_id': 'id', 'gene_name': 'name', 'gene_type': 'type'}, axis='columns')
    df_gdc = df_gdc.set_index('id')
    df_gdc = df_gdc.sort_index()

    # Read in and process the files to check our calculated results against (known FPKM and FPKM-UQ values)
    df_fpkm = pd.read_csv(os.path.join(links_dir, file_fpkm), sep='\t', names=['id','intensity'])
    df_fpkm_uq = pd.read_csv(os.path.join(links_dir, file_fpkm_uq), sep='\t', names=['id','intensity'])
    srs_known = [df_fpkm, df_fpkm_uq]
    for idf, df in enumerate(srs_known):
        df = df.set_index('id')
        srs_known[idf] = df.sort_index().iloc[:,0]

    # Check for column equality between the two reference datafiles
    for colname in ['name', 'seqname', 'start', 'end', 'strand', 'type']:
        log_fcn('Columns equal between the 2 reference files?', df_gdc[colname].equals(df_gencode_genes[colname]))

    # Check that the ID columns of all five dataframes are exactly the same
    df_samples = [calculated_counts] + srs_known
    dfs = df_samples + [df_gdc, df_gencode_genes]
    ndfs = len(dfs)
    import numpy as np
    for idf1 in range(ndfs-1):
        for idf2 in np.array(range(ndfs-1-idf1)) + idf1+1:
            df1 = dfs[idf1]
            df2 = dfs[idf2]
            log_fcn('ID columns the same in all 5 dataframes?', idf1, idf2, df1.index.equals(df2.index))

    # Show that we've reproduced what GDC calls the "exon_length" and what I'm assuming is probably the "aggregate_length" as well
    log_fcn('Correct calculation of exon_/aggregate_length?', df_gencode_genes['exon_length'].equals(df_gdc['exon_length']))

    # Show that using these exon lengths we have achieved adjusted counts that are proportional to the FPKM values
    tmp = df_samples[0] / df_gencode_genes['exon_length'] / df_samples[1]
    tmp = tmp[tmp.notnull()]
    log_fcn('Adjusted counts using the calculated exon lengths proportional to the FPKM values?', tmp.std()/tmp.mean()*100, (tmp-tmp.mean()).abs().max()/tmp.mean()*100)

    # log_fcn how well I reproduced the normalized values that I downloaded from the GDC data portal
    log_fcn('Maximum percent error in FPKM: {}'.format((fpkm-df_samples[1]).abs().max() / df_samples[1].mean() * 100))
    log_fcn('Maximum percent error in FPKM-UQ: {}'.format((fpkm_uq-df_samples[2]).abs().max() / df_samples[2].mean() * 100))


# Get a list of text files available for each sample in the links_dir
# This is no longer used
def get_files_per_sample(links_dir):
    # Sample call: files_per_sample = tc.get_files_per_sample('/data/BIDS-HPC/private/projects/dmi2/data/all_gene_expression_files_in_target/links')

    # Import relevant libraries
    import glob
    import os

    # Get a list of all files (with pathnames removed) in links_dir, except for the manifest file
    txt_files = set([ x.split('/')[-1] for x in glob.glob(os.path.join(links_dir,'*')) ]) - {'MANIFEST.txt'}

    # Get the corresponding sorted set of basenames (ostensibly, the unique sample names) from the file list
    basenames = sorted(set([ x.split('.')[0] for x in txt_files ]))

    # For each sample name, create a list of files having that sample name in the filename
    files_per_sample = []
    for basename in basenames:
        files_per_sample.append([ x for x in txt_files if basename in x ])

    # Return a list of text files available for each sample in the links_dir
    return(files_per_sample)


# Read in all the counts files and calculate the FPKM and FPKM-UQ values from them, checking the FPKM/FPKM-UQ values with known quantities if they're available
def get_intensities_old(files_per_sample, links_dir, df_gencode_genes, data_dir, nsamples=-1, log_fcn=print):

    # Import relevant libraries
    import os

    import pandas as pd
    tci = get_tci_library()

    # Constants (suffixes of the different file types in the links directory)
    # ['htseq.counts', 'htseq_counts.txt'] # these are the non-specified suffixes below
    star_counts_suffix = 'rna_seq.star_gene_counts.tsv'
    fpkm_suffix = 'FPKM.txt'
    fpkm_uq_suffix = 'FPKM-UQ.txt'

    # If the file containing the lists of series does not already exist...
    if not os.path.exists(os.path.join(data_dir,'series_lists.pkl')):

        # For the first namples_to_process samples in files_per_sample...
        srs_counts = []
        srs_fpkm = []
        srs_fpkm_uq = []
        nsamples = ( len(files_per_sample) if nsamples==-1 else nsamples )
        #for isample, files in enumerate(files_per_sample[:nsamples_to_process]):
        for isample, files in enumerate(files_per_sample[:nsamples]):

            # Initialize the descriptions of the sample (filenames namely) that we want to calculate
            counts_file = None
            fpkm_file = None
            fpkm_uq_file = None
            counts_type = None

            # For each file in the file list for the current sample...
            for ifile, x in enumerate([ curr_file.split('.')[1:] for curr_file in files ]):

                # Determine the suffix of the file
                suffix = '.'.join(x)

                # Run logic based on what the suffix of the current file is, calculating the scriptions of the sample (filenames namely) that we want to calculate
                if suffix == fpkm_suffix:
                    fpkm_file = files[ifile]
                elif suffix == fpkm_uq_suffix:
                    fpkm_uq_file = files[ifile]
                else:
                    if suffix == star_counts_suffix:
                        counts_type = 'STAR'
                    else:
                        counts_type = 'HTSeq'
                    counts_file = files[ifile]

            # Print the determined filenames and count filetype for the current sample
            # log_fcn('----')
            # log_fcn('Counts file ({}): {}'.format(counts_type, counts_file))
            # log_fcn('FPKM file: {}'.format(fpkm_file))
            # log_fcn('FPKM-UQ file: {}'.format(fpkm_uq_file))

            # Get counts dataframe for the current sample
            if counts_type == 'HTSeq':
                df_tmp = pd.read_csv(os.path.join(links_dir, counts_file), sep='\t', skipfooter=5, names=['id','intensity'])
            else:
                df_tmp = pd.read_csv(os.path.join(links_dir, counts_file), sep='\t', skiprows=5, usecols=[0,1], names=['id','intensity'])

            # Format the counts series and calculate FPKM and FPKM-UQ from it using the aggregate lengths in df_gencode_genes
            sr_counts = df_tmp.set_index('id').sort_index().iloc[:,0]
            sr_fpkm, sr_fpkm_uq = calculate_fpkm(df_gencode_genes, sr_counts)

            # Print how well I reproduced the FPKM values that I downloaded from the GDC data portal, if present
            if fpkm_file is not None:
                df_fpkm = pd.read_csv(os.path.join(links_dir, fpkm_file), sep='\t', names=['id','intensity'])
                sr_fpkm_known = df_fpkm.set_index('id').sort_index().iloc[:,0]
                perc_err = (sr_fpkm-sr_fpkm_known).abs().max() / sr_fpkm_known.mean() * 100
                #log_fcn('Maximum percent error in FPKM: {}'.format(perc_err))
                if perc_err > 1e-2:
                    log_fcn('ERROR: Maximum percent error ({}) in FPKM is too high!'.format(perc_err))
                    exit()

            # log_fcn how well I reproduced the FPKM-UQ values that I downloaded from the GDC data portal, if present
            if fpkm_uq_file is not None:
                df_fpkm_uq = pd.read_csv(os.path.join(links_dir, fpkm_uq_file), sep='\t', names=['id','intensity'])
                sr_fpkm_uq_known = df_fpkm_uq.set_index('id').sort_index().iloc[:,0]
                perc_err = (sr_fpkm_uq-sr_fpkm_uq_known).abs().max() / sr_fpkm_uq_known.mean() * 100
                #log_fcn('Maximum percent error in FPKM-UQ: {}'.format(perc_err))
                if perc_err > 1e-5:
                    log_fcn('ERROR: Maximum percent error ({}) in FPKM-UQ is too high!'.format(perc_err))
                    exit()

            # Append the current calculated series to the lists of series
            srs_counts.append(sr_counts)
            srs_fpkm.append(sr_fpkm)
            srs_fpkm_uq.append(sr_fpkm_uq)

            log_fcn('\r', '{:3.1f}% complete...'.format((isample+1)/nsamples*100), end='')

        # Write a pickle file containing the data that take a while to calculate
        tci.make_pickle([srs_counts, srs_fpkm, srs_fpkm_uq], data_dir, 'series_lists.pkl')

    # Otherwise, read it in
    else:
        [srs_counts, srs_fpkm, srs_fpkm_uq] = tci.load_pickle(data_dir, 'series_lists.pkl')

    # Return the calculated lists of series
    return(srs_counts, srs_fpkm, srs_fpkm_uq)


# Obtain a Pandas dataframe from the fields of interest for all samples, essentially containing everything we'll ever need to know about the samples, including the labels themselves
def get_labels_dataframe(sample_sheet_file, metadata_file, log_fcn=print):

    # Sample call:
    #   sample_sheet_file = '/data/BIDS-HPC/private/projects/dmi2/data/gdc_sample_sheet.2020-07-02.tsv'
    #   metadata_file = '/data/BIDS-HPC/private/projects/dmi2/data/metadata.cart.2020-07-02.json'
    #   df_samples = tc.get_labels_dataframe(sample_sheet_file, metadata_file)

    # Import relevant libraries
    import json
    import os

    import pandas as pd

    # Constants
    htseq_suffixes = ['htseq.counts', 'htseq_counts.txt']
    labels_df_names = ['sample id', 'file list index', 'counts file name', 'average base quality', 'file id', 'project id', 'case id', 'sample type', 'contamination_error', 'proportion_reads_mapped', 'proportion_reads_duplicated', 'contamination', 'proportion_base_mismatch', 'state', 'platform', 'average_read_length', 'entity_submitter_id']
    desired_keys = ['average_base_quality', 'contamination_error', 'proportion_reads_mapped', 'proportion_reads_duplicated', 'contamination', 'proportion_base_mismatch', 'state', 'platform', 'average_read_length']

    # Read in the two datafiles
    df_samples = pd.read_csv(sample_sheet_file, sep='\t')
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Get the corresponding filename mapping arrays
    filename_mapping_samples = df_samples['File Name']
    filename_mapping_metadata = []
    for curr_file in metadata:
        filename_mapping_metadata.append(curr_file['file_name'])

    # Get the full list of unique sample IDs
    samples = list(set(df_samples['Sample ID']))

    # For each unique sample ID...
    selected_fields_per_sample = []
    for sample in samples:

        # Store the dataframe for just the current sample
        df_sample = df_samples[df_samples['Sample ID']==sample]

        # Check that all relevant columns of the current sample are equal, as they should be since they all correspond to the same sample even though the rows correspond to different datafiles
        non_unique_values = (len(df_sample['Data Category'].unique())!=1) or (len(df_sample['Data Type'].unique())!=1) or (len(df_sample['Project ID'].unique())!=1) or (len(df_sample['Case ID'].unique())!=1) or (len(df_sample['Sample ID'].unique())!=1) or (len(df_sample['Sample Type'].unique())!=1)
        if non_unique_values:
            log_fcn('ERROR: All fields for the current sample are not equal over the files')
            exit()

        # Obtain the HTSeq counts files for the current sample (note: sometimes there are 2 instead of 1)
        htseq_files = [ (fn if ('.'.join(fn.split('.')[1:-1]) in htseq_suffixes) else None) for fn in df_sample['File Name'] ]
        counts_file_list = list(set(htseq_files) - {None})

        # If there is just a single HTSeq counts files for the current sample, then we have already identified the HTSeq counts file that we're looking for
        if len(counts_file_list) == 1:
            counts_file = counts_file_list[0]

        # If there are more than one HTSeq counts files for the current sample, it doesn't make sense to me to keep multiple analyses of the same sample, so just choose the analysis with the best base quality score (or the first if multiple files have the same best score)
        else:

            # Initialize variables in this small loop over possible HTSeq counts files
            best_score = -1
            best_counts_file = None

            # For each counts file...
            for counts_file in counts_file_list:

                # Obtain the corresponding index of the metadata list, using that to extract the average_base_quality field
                metadata_index = filename_mapping_metadata.index(counts_file)
                score = metadata[metadata_index]['analysis']['input_files'][0]['average_base_quality']

                # If the current score is better than the current best score, update the variables
                if score > best_score:
                    best_counts_file = counts_file
                    best_score = score

            # Rename the variable holding the HTSeq counts file that we were looking for
            counts_file = best_counts_file

        # For the best HTSeq counts file for the current sample, obtain the corresponding index of in the sample sheet and in the metadata, and check that they are the same
        samples_index = filename_mapping_samples[filename_mapping_samples==counts_file].index[0]
        metadata_index = filename_mapping_metadata.index(counts_file)
        if samples_index != metadata_index:
            log_fcn('ERROR: The File indexes for the sample sheet and the metadata file are different')
            exit()

        # Get shortcut variables for the sections of the sample sheet and the metadata that have some values that we want to add to our labels dataframe
        series = df_samples.loc[samples_index,:]
        md1 = metadata[metadata_index]['analysis']['input_files'][0]

        # Run a check to ensure at least a None value of all desired keys exists in the md1 dictionary
        current_keys = md1.keys()
        for desired_key in desired_keys:
            if desired_key not in current_keys:
                md1[desired_key] = None

        # Save the fields of interest to a running list
        selected_fields_per_sample.append([sample, samples_index, counts_file, md1['average_base_quality'], series['File ID'], series['Project ID'], series['Case ID'], series['Sample Type'], md1['contamination_error'], md1['proportion_reads_mapped'], md1['proportion_reads_duplicated'], md1['contamination'], md1['proportion_base_mismatch'], md1['state'], md1['platform'], md1['average_read_length'], metadata[metadata_index]['associated_entities'][0]['entity_submitter_id']])

    # Define the Pandas dataframe from the fields of interest for all samples, finishing by sorting by index
    df = pd.DataFrame(data=selected_fields_per_sample, columns=labels_df_names)
    df = df.set_index('sample id').sort_index()

    # Return this dataframe
    return(df)


# Used to remove outliers from analysis
higher_is_better_by_column = {
    'average base quality': True, 
    'proportion_base_mismatch': False, 
    'proportion_reads_mapped': True
}

# Plot histograms of the numerical columns of the samples/labels before and after cutoffs could theoretically be applied, and print out a summary of what we should probably do
def remove_bad_samples(df_samples, nstd_by_column, log_fcn=print):

    # Import relevant library
    import numpy as np

    # Start off filtering none of the data
    valid_ind = np.full((len(df_samples),), True)

    for column, nstd in nstd_by_column.items():
        vals = df_samples[column]
        higher_is_better = higher_is_better_by_column[column]
        # Determine +1 or -1 depending on whether higher is better (if higher is better, use -1)
        sign = -2*int(higher_is_better) + 1

        # Calculate the cutoff for the current plot using the inputted number of standard deviations from the mean as the cutoff
        cutoff = vals.mean() + sign*nstd*vals.std()

        # Determine a boolean array of where the values fall outside the cutoffs and log_fcn how many such bad values there are
        bad_vals = (sign*vals) > (sign*cutoff)
        log_fcn('There are {} bad values in the "{}" plot'.format(sum(bad_vals), column))

        # Update the boolean filtering array using the current set of bad values
        valid_ind[bad_vals]=False

    return(df_samples.iloc[valid_ind,:], valid_ind)


# Return the samples dataframe with the samples removed that correspond to multiple cases (i.e., people)
def drop_multiperson_samples(df_samples, log_fcn=print):

    # Import relevant libraries
    import numpy as np
    import pandas as pd

    # Initialize the arrays of interest
    indexes_to_drop = [] # to store indexes of samples to drop
    samples_to_drop = [] # to store the samples themselves (Series) to drop
    indexes_to_keep = np.full((len(df_samples),), True) # to store the indexes to keep in the full samples array so that I can plug these logical indexes into other arrays

    # For every index in the sample dataframe...
    for isample, sample_index in enumerate(df_samples.index):

        # Save the current sample (as a Series)
        sample = df_samples.loc[sample_index]

        # If there are multiple cases (people; and I've confirmed that they can't be the same people, e.g., living white female and dead black male) corresponding to this one sample...
        if len(sample['case id'].split()) > 1:
            indexes_to_drop.append(sample_index) # save the sample index
            samples_to_drop.append(sample) # save the sample series
            indexes_to_keep[isample] = False

    # Create and print a Pandas dataframe of the samples to drop in order to visualize it nicely
    log_fcn('Dropping the following samples from the samples table:')
    df_samples_to_drop = pd.DataFrame(data=samples_to_drop).rename_axis(index='sample id')
    log_fcn(df_samples_to_drop)

    # Return the modified samples dataframe
    return(df_samples.drop(index=indexes_to_drop), indexes_to_keep, df_samples_to_drop)


# Perform exploratory data analysis on the sample labels
def eda_labels(df_samples, log_fcn=print, plt_ctx=None):

    # Import relevant library
    import random

    import matplotlib.pyplot as plt

    import contextlib

    if not plt_ctx:
        plt_ctx = contextlib.nullcontext()

    # Add the index "column" as an actual column to the dataframe so we can analyze the index column in the same manner as the other columns
    df_samples[df_samples.index.name] = df_samples.index

    # Initialize the holder lists of the column types
    cols_unique = []
    cols_uniform = []
    cols_other = []

    # Get the total number of rows in the dataframe
    nsamples = len(df_samples)

    # Get a random index in the dataframe
    rand_index = random.randrange(nsamples)

    # Plot histograms of the numeric data
    with plt_ctx:
        fig, ax = plt.subplots(figsize=(12,8), facecolor='w')
        _ = df_samples.hist(ax=ax)
        plt.show(fig)

    # Determine the non-numeric columns
    non_numeric_cols = df_samples.select_dtypes(exclude='number').columns

    # Initialize the column name lengths
    max_col_len = -1

    # For every non-numeric column...
    for col in non_numeric_cols:

        # Determine the number of unique values in the column
        nunique = len(df_samples[col].unique())

        # Every row in the column is unique
        if nunique == nsamples:
            cols_unique.append([col, df_samples[col][rand_index]])

        # The column is completely uniform
        elif nunique == 1:
            cols_uniform.append([col, df_samples[col][0]])

        # The column is neither unique nor uniform
        else:
            cols_other.append([col, nunique])

        # Possibly update the maximum column name size (for pretty printing later)
        if len(col) > max_col_len:
            max_col_len = len(col)

    # Store the output format string depending on the maximum column name size
    output_col_str = ' . {:' + str(max_col_len+5) + '}{}'

    # Print the columns in which every row is unique
    log_fcn('Non-numeric columns with all unique values ({} of them), with sample values:\n'.format(nsamples))
    for col_data in cols_unique:
        log_fcn(output_col_str.format(col_data[0], col_data[1]))

    # Print the columns that are completely uniform
    log_fcn('\nNon-numeric columns with uniform values:\n')
    for col_data in cols_uniform:
        log_fcn(output_col_str.format(col_data[0], col_data[1]))

    # Print the columns (and supporting information) that are neither unique nor uniform
    log_fcn('\nNon-numeric columns with non-unique and non-uniform values:\n')
    for col_data in cols_other:
        log_fcn(output_col_str.format(col_data[0], col_data[1]), '\n')
        log_fcn(df_samples[col_data[0]].value_counts(), '\n')


# Read in the counts for all the samples in the samples dataframe df_samples
# the counts dataframe will be in the same order as df_samples
def get_counts(df_samples, links_dir, log_fcn=print):

    # Import relevant libraries
    import os

    import pandas as pd

    # Ensure that all values in the "counts file name" column of df_samples are unique as expected
    nsamples = len(df_samples)
    if not len(df_samples['counts file name'].unique()) == nsamples:
        log_fcn('ERROR: "counts file name" column of the samples dataframe does not contain all unique values')
        exit()

    # Strip the ".gz" off of the filenames in the "counts file name" column of the samples dataframe
    counts_filenames = [ x.split(sep='.gz')[0] for x in df_samples['counts file name'] ]

    # For every counts filename in the samples dataframe...
    srs_counts = []
    for isample, counts_fn in enumerate(counts_filenames):

        # Read in the counts data
        sr_counts = pd.read_csv(os.path.join(links_dir, counts_fn), sep='\t', skipfooter=5, names=['id','intensity']).set_index('id').sort_index().iloc[:,0] # assume this is of the HTSeq (as opposed to STAR) format

        # Append the read-in and calculated values to running lists
        srs_counts.append(sr_counts)

        log_fcn('\r', '{:3.1f}% complete...'.format((isample+1)/nsamples*100), end='')

    # Put the list of series into a Pandas dataframe
    df_counts = pd.DataFrame(srs_counts, index=df_samples.index)

    # Return the calculated lists of series
    return(df_counts)


# Convert the lists of Pandas series to Pandas dataframes
def make_intensities_dataframes(srs_list, index):
    import pandas as pd
    counts_list = []
    for srs in srs_list:
        counts_list.append(pd.DataFrame(srs, index=index))
    return(counts_list)


# Print some random data for us to spot-check in the files themselves to manually ensure we have a handle on the data arrays
def spot_check_data(df_samples, df_counts, df_fpkm, df_fpkm_uq, nsamples=4, log_fcn=print):

    # Import relevant library
    import random

    # Constants
    intensity_types = ['counts', 'FPKM', 'FPKM-UQ']

    # Variable
    intensities = [df_counts, df_fpkm, df_fpkm_uq]

    # Get some values from the intensity data
    nsamples_tot = intensities[0].shape[0]
    sample_names = intensities[0].index

    # For each intensity type...
    for iintensity, intensity_type in enumerate(intensity_types):

        # For each of nsamples random samples in the data...
        for sample_index in random.sample(range(nsamples_tot), k=nsamples):

            # Store the current sample name
            sample_name = sample_names[sample_index]

            # Get the non-zero intensities for the current sample
            srs = intensities[iintensity].iloc[sample_index,:]
            srs2 = srs[srs!=0]

            # Get a random index of the non-zero intensities and store the corresponding intensity and gene
            srs2_index = random.randrange(len(srs2))
            intensity = srs2[srs2_index]
            gene = srs2.index[srs2_index]

            # Get some important data from the samples dataframe
            project_id = df_samples.loc[sample_name, 'project id']
            sample_type = df_samples.iloc[sample_index, 6]

            # Print what we should see in the files
            log_fcn('Sample {} ({}, {}) should have a {} value of {} for gene {}'.format(sample_name, project_id, sample_type, intensity_type, intensity, gene))


# Load the data downloaded from the GDC Data Portal
def load_gdc_data(sample_sheet_file, metadata_file, links_dir, log_fcn=print):

    # Import the relevant libraries
    import os
    tci = get_tci_library()

    # Identify the data directory as the directory that the sample sheet file is in
    os_sep = os.sep
    data_dir = os_sep.join(sample_sheet_file.split(sep=os_sep)[:-1])

    # If the file containing the GDC data does not already exist...
    if not os.path.exists(os.path.join(data_dir,'gdc_data.pkl')):

        # Obtain a Pandas dataframe from the fields of interest for all samples, essentially containing everything we'll ever need to know about the samples, including the labels themselves
        df_samples = get_labels_dataframe(sample_sheet_file, metadata_file) # this will always be in alphabetical order of the sample IDs

        # Read in the counts for all the samples in the samples dataframe df_samples
        df_counts = get_counts(df_samples, links_dir) # the counts dataframe will be in the same order as df_samples

        # Write a pickle file containing the data that takes a while to calculate
        tci.make_pickle([df_samples, df_counts], data_dir, 'gdc_data.pkl')

    # Otherwise, read it in
    else:
        [df_samples, df_counts] = tci.load_pickle(data_dir, 'gdc_data.pkl', log_fcn=log_fcn)

    return(df_samples, df_counts)


# Calculate the FPKM and FPKM-UQ dataframes, and check them with known values if the needed datafiles are present
# the FPKM and FPKM-UQ dataframes will be in the same order as df_samples
# since df_counts is in the same order as df_samples, then the counts and FPKM/FPKM-UQ dataframes will also be aligned
# regardless, this shouldn't be a problem moving forward, since df_samples will always be in lexical order of the sample IDs!
# not to mention, whenever we're unsure, we should run the spot-checks!
def get_fpkm(df_counts, annotation_file, df_samples, links_dir, log_fcn=print):

    # Import relevant libraries
    import os

    import pandas as pd
    tci = get_tci_library()

    # Identify the data directory as the directory that the annotation file is in
    os_sep = os.sep
    data_dir = os_sep.join(annotation_file.split(sep=os_sep)[:-1])

    # If the file containing the FPKM/FPKM-UQ data does not already exist...
    if not os.path.exists(os.path.join(data_dir,'fpkm_data.pkl')):

        # Ensure that all values in the "counts file name" column of df_samples are unique as expected
        nsamples = len(df_samples)

        # Strip the ".gz" off of the filenames in the "counts file name" column of the samples dataframe
        counts_filenames = [ x.split(sep='.gz')[0] for x in df_samples['counts file name'] ]

        # Obtain a listing of all the files in the links directory
        files_in_links_dir = os.listdir(links_dir)

        # Prepare the annotation dataframe df_gencode_genes, particularly calculating the exon length of each gene (corresponding to its non-overlapping exons) and adding this as a column to the df_gencode_genes dataframe
        # This takes about 10 minutes if the pickle file doesn't already exist
        df_gencode_genes = calculate_exon_lengths(annotation_file)

        # For every counts filename in the samples dataframe...
        srs_fpkm = []
        srs_fpkm_uq = []
        for isample, counts_fn in enumerate(counts_filenames):

            # Read in the counts data
            sr_counts = df_counts.iloc[isample,:]

            # Use those counts data to calculate the FPKM and FPKM-UQ values
            sr_fpkm, sr_fpkm_uq = calculate_fpkm(df_gencode_genes, sr_counts)

            # Get the basename of the current counts file
            bn = counts_fn.split(sep='.')[0]

            # Determine the files in files_in_links_dir and their indexes matching the current basename
            bn_matches = []
            bn_matches_indexes = []
            for ifile, curr_file in enumerate(files_in_links_dir):
                if bn in curr_file:
                    bn_matches.append(curr_file)
                    bn_matches_indexes.append(ifile)

            # From the matching files, determine their suffixes in lowercase, finding where in them FPKM and FPKM-UQ strings match
            suffixes_lower = [ x.split(sep=bn+'.')[1].lower() for x in bn_matches ]
            fpkm_matches = [ 'fpkm.' in x for x in suffixes_lower ]
            fpkm_uq_matches = [ 'fpkm-uq.' in x for x in suffixes_lower ]

            # Ensure there aren't more than 1 match for either FPKM or FPKM-UQ for the current basename
            if sum(fpkm_matches)>1 or sum(fpkm_uq_matches)>1:
                log_fcn('ERROR: More than 1 FPKM or FPKM-UQ file matches the basename {}'.format(bn))
                exit()

            # If an FPKM file corresponding to the current basename is found...
            if sum(fpkm_matches) == 1:

                # Determine its filename
                fpkm_fn = files_in_links_dir[bn_matches_indexes[fpkm_matches.index(True)]]

                # Read in its data into a Pandas series
                sr_fpkm_known = pd.read_csv(os.path.join(links_dir, fpkm_fn), sep='\t', names=['id','intensity']).set_index('id').sort_index().iloc[:,0]

                # Determine how well our calculated values in sr_fpkm match those read in to sr_fpkm_known
                perc_err = (sr_fpkm-sr_fpkm_known).abs().max() / sr_fpkm_known.mean() * 100
                if perc_err > 1e-2:
                    log_fcn('ERROR: Maximum percent error ({}) in FPKM is too high!'.format(perc_err))
                    exit()

            # If an FPKM-UQ file corresponding to the current basename is found...
            if sum(fpkm_uq_matches) == 1:

                # Determine its filename
                fpkm_uq_fn = files_in_links_dir[bn_matches_indexes[fpkm_uq_matches.index(True)]]

                # Read in its data into a Pandas series
                sr_fpkm_uq_known = pd.read_csv(os.path.join(links_dir, fpkm_uq_fn), sep='\t', names=['id','intensity']).set_index('id').sort_index().iloc[:,0]

                # Determine how well our calculated values in sr_fpkm_uq match those read in to sr_fpkm_uq_known
                perc_err = (sr_fpkm_uq-sr_fpkm_uq_known).abs().max() / sr_fpkm_uq_known.mean() * 100
                if perc_err > 1e-5:
                    log_fcn('ERROR: Maximum percent error ({}) in FPKM-UQ is too high!'.format(perc_err))
                    exit()

            # Append the read-in and calculated values to running lists
            srs_fpkm.append(sr_fpkm)
            srs_fpkm_uq.append(sr_fpkm_uq)

            log_fcn('\r', '{:3.1f}% complete...'.format((isample+1)/nsamples*100), end='')

        # Put the lists of series into dataframes
        df_fpkm = pd.DataFrame(srs_fpkm, index=df_samples.index)
        df_fpkm_uq = pd.DataFrame(srs_fpkm_uq, index=df_samples.index)

        # Write a pickle file containing the data that takes a while to calculate
        tci.make_pickle([df_fpkm, df_fpkm_uq], data_dir, 'fpkm_data.pkl')

    # Otherwise, read it in
    else:
        [df_fpkm, df_fpkm_uq] = tci.load_pickle(data_dir, 'fpkm_data.pkl', log_fcn=log_fcn)

    return(df_fpkm, df_fpkm_uq)


# Calculate the TPM using the counts and gene lengths
def get_tpm(C_df, annotation_file, log_fcn=print):

    # Note: I've confirmed TPM calculation with get_tpm_from_fpkm() function below using both FPKM and FPKM-UQ via:
    # df_tpm = tc.get_tpm(df_counts, annotation_file)
    # df_tpm1 = tc.get_tpm_from_fpkm(df_fpkm)
    # df_tpm2 = tc.get_tpm_from_fpkm(df_fpkm_uq)
    # import numpy as np
    # log_fcn(np.amax(np.abs(df_tpm1-df_tpm).to_numpy(), axis=(0,1)))
    # log_fcn(np.amax(np.abs(df_tpm2-df_tpm).to_numpy(), axis=(0,1)))
    # log_fcn(np.amax(np.abs(df_tpm2-df_tpm1).to_numpy(), axis=(0,1)))
    # log_fcn(np.sqrt(np.mean(((df_tpm1-df_tpm)**2).to_numpy(), axis=(0,1))))
    # log_fcn(np.sqrt(np.mean(((df_tpm2-df_tpm)**2).to_numpy(), axis=(0,1))))
    # log_fcn(np.sqrt(np.mean(((df_tpm2-df_tpm1)**2).to_numpy(), axis=(0,1))))

    # Import relevant library
    import numpy as np

    # Calculate the aggregate exon lengths the way GDC does it
    # series of length ngenes
    L_srs = calculate_exon_lengths(annotation_file, log_fcn=log_fcn)['exon_length']

    # Ensure the gene order in the counts and lengths is consistent so that we can perform joint operations on them
    if not C_df.columns.equals(L_srs.index):
        log_fcn('ERROR: Order of genes in the counts dataframe is not the same as that in the lengths series')
        exit()

    # Extract the numbers of samples and genes; it seems like this may be unnecessary as seen in the comment after counts_norm, but doing things explicitly like this is significantly faster
    # C_df is a dataframe of shape (nsamples, ngenes)
    nsamples, ngenes = C_df.shape

    # Normalize the counts by their corresponding gene lengths
    # denominator: (nsamples,ngenes) --> repeats over axis=0 (i.e., depends only on gene, not sample) --> L_ij = L_j
    # numerator: (nsamples,ngenes) --> C_ij
    # Cn_ij = C_ij / L_j
    counts_norm = C_df / np.tile(np.expand_dims(L_srs, axis=0), (nsamples,1)) # this equals "C_df / L_srs" (which is simpler) but doing it this way is significantly faster

    # Calculate the normalization factor for each sample
    # D_ij = SUM(Cn_ij,j) --> repeats over axis=1 (i.e., depends only on sample, not gene) --> D_ij = D_i
    denom = np.tile(np.expand_dims(counts_norm.sum(axis=1), axis=1), (1,ngenes))

    # Calculate the TPM
    # Cn_ij / D_ij * 10^6
    # C_ij / L_j / SUM(C_ij/L_j,j) * 10^6
    # T_si = C_si / L_i / SUM(C_sk/L_k,k) * 10^6
    # This is perfectly consistent with boxed formula in sectino 2.3.1 of tpm_calculation.pdf
    tpm = counts_norm / denom * 1e6

    return(tpm)


# Calculate the TPM using FPKM or FPKM-UQ
def get_tpm_from_fpkm(F_df):

    # Import relevant library
    import numpy as np

    # Extract the number of genes
    # F_ij: (nsamples,ngenes)
    ngenes = F_df.shape[1]

    # Calculate the normalization factor for each sapmle
    # D_ij = SUM(F_ij,j) --> repeats over axis=1 (i.e., depends only on sample, not gene) --> D_ij = D_i
    denom = np.tile(np.expand_dims(F_df.sum(axis=1), axis=1), (1,ngenes))

    # Calculate the TPM
    # T_ij = F_ij / D_ij * 10^6
    # T_si = F_si / SUM(F_sk,k) * 10^6
    # This formula is perfectly consistent with the first lines in sections 2.3.1 and 2.3.2 of tpm_calculation.pdf
    tpm = F_df / denom * 1e6

    return(tpm)


# Write annotation and gene counts files (two files total) that are in the same format as the pasilla example so that we can follow the steps outlined at
# http://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html#count-matrix-input
def write_sample_for_deseq2_input(srs_labels, df_counts, data_directory, dataset_name, reqd_string_in_label='primary tumor', nsamples_per_condition=[5,3,9], log_fcn=print):

    # Sample call: write_sample_for_deseq2_input(df_samples['label 1'], df_counts, data_directory)

    # Import relevant libraries
    import os
    import random

    import numpy as np
    tci = get_tci_library()

    # Get a subset (using both a string in the condition names and a particular number of conditions) of the series of the value counts of the label of interest
    label_value_counts = srs_labels.value_counts()
    srs_subset = label_value_counts[[ reqd_string_in_label in x.lower() for x in label_value_counts.index ]][:len(nsamples_per_condition)]
    log_fcn('Using the following conditions (though not all of the samples for each label):')
    log_fcn(srs_subset)

    # Construct a list of indexes (actual numbers) to use as a sample of all our data
    all_indexes_to_use = []
    for label, nsamples in zip(srs_subset.index, nsamples_per_condition): # for each condition (label) and inputted number of samples to use for each condition...
        indexes = np.argwhere((srs_labels==label).to_numpy()).flatten() # get the numerical indexes of the current label
        #indexes_to_use = list(indexes[:nsamples]) # get just the number of numerical indexes that we want for the current condition - first nsamples of the list
        indexes_to_use = random.sample(list(indexes), nsamples) # get just the number of numerical indexes that we want for the current condition - random nsamples of the list
        log_fcn('\nHere are the {} indexes out of {} that correspond to the condition {}:'.format(len(indexes), len(srs_labels), label))
        log_fcn(indexes)
        #log_fcn('However, we\'re only using the first {}:'.format(nsamples))
        log_fcn('However, we\'re using just a random sample of {} items:'.format(nsamples))
        log_fcn(indexes_to_use)
        all_indexes_to_use = all_indexes_to_use + indexes_to_use
    log_fcn('\nHere is the final set of numerical indexes that we\'re using ({}={} of them):'.format(sum(nsamples_per_condition), len(all_indexes_to_use)))
    log_fcn(all_indexes_to_use)

    # Get just a sample of the labels/conditions and counts
    all_samples_to_use = srs_labels.index[all_indexes_to_use] # get the actual descriptive indexes from the numerical indexes
    labels_to_use = srs_labels[all_samples_to_use]
    counts_to_use = df_counts.loc[all_samples_to_use,:].transpose()

    # Delete rows of counts that are all zeros
    counts_to_use = counts_to_use[(counts_to_use!=0).any(axis=1)]

    # Do a quick check of the list of labels/conditions
    conditions_list = []
    for nsamples, label in zip(nsamples_per_condition, labels_to_use[np.cumsum(nsamples_per_condition)-1]):
        conditions_list = conditions_list + [label]*nsamples
    if conditions_list != labels_to_use.to_list():
        log_fcn('ERROR: The actual list of labels/conditions is not what\'s expected')
        exit()

    # Check that the indexes of the counts and labels that we're going to write out are the same
    if not counts_to_use.columns.equals(labels_to_use.index):
        log_fcn('ERROR: Indexes/columns of the labels/counts are inconsistent')
        exit()

    # Create the dataset directory if it doesn't already exist
    #os.makedirs(os.path.join(data_directory, 'datasets', dataset_name))
    os.makedirs(os.path.join(data_directory, 'datasets', dataset_name), exist_ok=True)

    # Write the annotation file in the same format as the pasilla example
    with open(file=os.path.join(data_directory, 'datasets', dataset_name, 'annotation.csv'), mode='w') as f:
        log_fcn('"file","condition"', file=f)
        for curr_file, condition in zip(labels_to_use.index, labels_to_use):
            log_fcn('"{}","{}"'.format(curr_file, condition), file=f)

    # Write the gene counts in the same format as the pasilla example
    with open(file=os.path.join(data_directory, 'datasets', dataset_name, 'gene_counts.tsv'), mode='w') as f:
        counts_to_use.to_csv(f, sep='\t', index_label='gene_id')

    # Save the dataset data
    tci.make_pickle([srs_labels, df_counts, data_directory, dataset_name, reqd_string_in_label, nsamples_per_condition, label_value_counts, srs_subset, all_indexes_to_use, all_samples_to_use, labels_to_use, counts_to_use, conditions_list], os.path.join(data_directory, 'datasets', dataset_name), dataset_name+'.pkl')


# Write all the data for input into DESeq2, instead of just a sample
def write_all_data_for_deseq2_input(srs_labels, df_counts, data_directory, dataset_name, drop_zero_genes=False, log_fcn=print):

    # Sample call: write_all_data_for_deseq2_input(df_samples['label 1'], df_counts, data_directory, 'all_data')

    # Import relevant libraries
    import os

    import numpy as np
    tci = get_tci_library()

    # Construct a list of indexes (actual numbers) to use as a sample of all our data (this time we're using them all)
    all_indexes_to_use = [ x for x in range(len(srs_labels)) ]
    log_fcn('\nHere is the final set of numerical indexes that we\'re using ({} of them):'.format(len(all_indexes_to_use)))
    log_fcn(all_indexes_to_use)

    # Get just a sample of the labels/conditions and counts
    all_samples_to_use = srs_labels.index[all_indexes_to_use] # get the actual descriptive indexes from the numerical indexes
    labels_to_use = srs_labels[all_samples_to_use]
    counts_to_use = df_counts.loc[all_samples_to_use,:].transpose()

    # Delete rows of counts that are all zeros
    if drop_zero_genes:
        counts_to_use = counts_to_use[(counts_to_use!=0).any(axis=1)]

    # Check that the indexes of the counts and labels that we're going to write out are the same
    if not counts_to_use.columns.equals(labels_to_use.index):
        log_fcn('ERROR: Indexes/columns of the labels/counts are inconsistent')
        exit()

    # Create the dataset directory if it doesn't already exist
    #os.makedirs(os.path.join(data_directory, 'datasets', dataset_name))
    os.makedirs(os.path.join(data_directory, 'datasets', dataset_name), exist_ok=True)

    # Write the annotation file in the same format as the pasilla example
    with open(file=os.path.join(data_directory, 'datasets', dataset_name, 'annotation.csv'), mode='w') as f:
        log_fcn('"file","condition"', file=f)
        for curr_file, condition in zip(labels_to_use.index, labels_to_use):
            log_fcn('"{}","{}"'.format(curr_file, condition), file=f)

    # Write the gene counts in the same format as the pasilla example
    with open(file=os.path.join(data_directory, 'datasets', dataset_name, 'gene_counts.tsv'), mode='w') as f:
        counts_to_use.to_csv(f, sep='\t', index_label='gene_id')

    # Save the dataset data
    tci.make_pickle([srs_labels, df_counts, data_directory, dataset_name, all_indexes_to_use, all_samples_to_use, labels_to_use, counts_to_use], os.path.join(data_directory, 'datasets', dataset_name), dataset_name+'.pkl')


# Create and plot PCA and tSNE analyses
def plot_pca_and_tsne(data_directory, dataset_name, transformation_name='variance-stabilizing', ntop=500, n_components_pca=10, alpha=1, dpi=300, y=None, save_figure=False, log_fcn=print, plt_ctx=None):

    # Sample call: plot_pca_and_tsne('/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data/assay_normal_transformation.csv', '/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data/coldata_normal_transformation.csv', 'normal', '/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data')

    # Import relevant libraries
    import os

    import matplotlib.lines as mpl_lines
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import sklearn.decomposition as sk_decomp
    import sklearn.manifold as sk_manif

    import contextlib

    if not plt_ctx:
        plt_ctx = contextlib.nullcontext()

    # Process the arguments
    transformation_name_filename = transformation_name.lower().replace(' ','_').replace('-','_') # get a version of the transformation_name suitable for filenames
    data_dir = os.path.join(data_directory, 'datasets', dataset_name) # '/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data'
    assay_csv_file = os.path.join(data_dir, 'assay_' + transformation_name_filename + '_transformation.csv') # '/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data/assay_variance_stabilizing_transformation.csv'
    coldata_csv_file = os.path.join(data_dir, 'coldata_' + transformation_name_filename + '_transformation.csv') # '/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data/coldata_variance_stabilizing_transformation.csv'

    # Determine the data matrix
    df_assay = pd.read_csv(assay_csv_file).set_index('Unnamed: 0') # read in the transformed data
    top_genes = df_assay.var(axis=1).sort_values(axis=0, ascending=False)[:ntop].index # get the indexes of the top-ntop-variance genes
    X = df_assay.loc[top_genes,:].T # keep only the top genes and transpose in order to get the typical data matrix format with the samples in the rows

    # Determine the labels vector
    if y is None:
        df_coldata = pd.read_csv(coldata_csv_file).set_index('Unnamed: 0') # read in the column data, which includes the labels (in the 'condition' column)
        y = df_coldata.loc[X.index,'condition'] # ensure the labels are ordered in the same way as the data and take just the 'condition' column as the label
        fn_addendum = ''
    else: # allow for a custom set of labels
        y = y.loc[X.index]
        fn_addendum = '_custom_label'

    # Order the samples by their labels
    sample_order = y.sort_values().index
    y = y.loc[sample_order]
    X = X.loc[sample_order,:]
    if not y.index.equals(X.index):
        log_fcn('ERROR: Weirdly inconsistent ordering')
        exit()

    # Perform PCA
    pca = sk_decomp.PCA(n_components=n_components_pca)
    pca_res = pca.fit_transform(X)
    log_fcn('Top {} PCA explained variance ratios: {}'.format(n_components_pca, pca.explained_variance_ratio_))

    # Get a reasonable set of markers and color palette
    markers = mpl_lines.Line2D.filled_markers
    nclasses = len(set(y))
    marker_list = markers * int(nclasses/len(markers)+1)
    color_palette = sns.color_palette("hls", nclasses)

    # Plot and save the PCA
    with plt_ctx:
        fig = plt.figure(figsize=(12,7.5), facecolor='w')
        ax = sns.scatterplot(x=pca_res[:,0], y=pca_res[:,1], hue=y, style=y, palette=color_palette, legend="full", alpha=alpha, markers=marker_list, edgecolor='k')
        ax.legend(bbox_to_anchor=(1,1))
        ax.set_title('PCA - ' + transformation_name + ' transformation')
        if save_figure:
            fig.savefig(os.path.join(data_dir, 'pca_' + transformation_name_filename + '_transformation' + fn_addendum + '.png'), dpi=dpi, bbox_inches='tight')
        plt.show(fig)

    # Perform tSNE analysis
    tsne = sk_manif.TSNE(n_components=2)
    tsne_res = tsne.fit_transform(X)

    # Plot and save the tSNE analysis
    with plt_ctx:
        fig = plt.figure(figsize=(12,7.5), facecolor='w')
        ax = sns.scatterplot(x=tsne_res[:,0], y=tsne_res[:,1], hue=y, style=y, palette=color_palette, legend="full", alpha=alpha, markers=marker_list, edgecolor='k')
        ax.legend(bbox_to_anchor=(1,1))
        ax.set_title('tSNE - ' + transformation_name + ' transformation')
        if save_figure:
            fig.savefig(os.path.join(data_dir, 'tsne_' + transformation_name_filename + '_transformation' + fn_addendum + '.png'), dpi=dpi, bbox_inches='tight')
        plt.show(fig)


# Run VST using DESeq2 on data exported from Python
def run_deseq2(dataset_name, project_directory):
    # run_deseq2('all_data_label_2', project_directory)
    import os
    import subprocess
#    cmd_list = ['Rscript', '--vanilla', os.path.join(project_directory, 'checkout', 'run_vst.R'), dataset_name, project_directory]
#    log_fcn('Now running command: ' + ' '.join(cmd_list))
#    list_files = subprocess.run(cmd_list)
#    log_fcn('The Rscript exit code was {}'.format(list_files.returncode))


# This function will take the raw counts and their labels and return the data matrix X (dataframe) and labels vector y (series) with the samples in label order and the genes in top-variance order by running the VST using DESeq2, saving all intermediate files
def run_vst(counts_dataframe, labels_series, project_directory, log_fcn=print):

    # Sample call: X, y = run_vst(df_counts, df_samples['label 1'], project_directory)

    # Import relevant libraries
    import os

    import pandas as pd
    tci = get_tci_library()

    # Constant (basically)
    transformation_name = 'variance-stabilizing'

    # Variables
    dataset_name = labels_series.name.lower().replace(' ','_').replace('-','_')
    data_dir = os.path.join(project_directory, 'data')
    transformation_name_filename = transformation_name.lower().replace(' ','_').replace('-','_') # get a version of the transformation_name suitable for filenames
    data_dir2 = os.path.join(data_dir, 'datasets', dataset_name)
    assay_csv_file = os.path.join(data_dir2, 'assay_' + transformation_name_filename + '_transformation.csv') # '/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data/assay_variance_stabilizing_transformation.csv'
    coldata_csv_file = os.path.join(data_dir2, 'coldata_' + transformation_name_filename + '_transformation.csv') # '/data/BIDS-HPC/private/projects/dmi2/data/datasets/all_data/coldata_variance_stabilizing_transformation.csv'

    # If the datafile does not already exist...
    if not os.path.exists(os.path.join(data_dir2, 'vst_transformed_data.pkl')):

        # Write all the data for input into DESeq2
        write_all_data_for_deseq2_input(labels_series, counts_dataframe, data_dir, dataset_name)

        # Run VST using DESeq2 on the dataset exported from Python above
        # Note this will write (at least) the files assay_csv_file and coldata_csv_file defined above
        run_deseq2(dataset_name, project_directory)

        # Determine the data matrix
        df_assay = pd.read_csv(assay_csv_file).set_index('Unnamed: 0') # read in the transformed data
        top_variance_order = df_assay.var(axis=1).sort_values(axis=0, ascending=False).index # get the indexes of the genes in top-variance order
        X = df_assay.loc[top_variance_order,:].T # order the genes by top variance and transpose in order to get the typical data matrix format with the samples in the rows

        # Determine the labels vector
        df_coldata = pd.read_csv(coldata_csv_file).set_index('Unnamed: 0') # read in the column data, which includes the labels (in the 'condition' column)
        y = df_coldata.loc[X.index,'condition'] # ensure the labels are ordered in the same way as the data and take just the 'condition' column as the label

        # Order the samples by their labels
        sample_order = y.sort_values().index
        y = y.loc[sample_order]
        X = X.loc[sample_order,:]

        # This should be a trivial check
        if not y.index.equals(X.index):
            log_fcn('ERROR: Weirdly inconsistent ordering')
            exit()

        # Save the data to disk
        tci.make_pickle([X, y], data_dir2, 'vst_transformed_data.pkl')

    # Otherwise, read it in
    else:
        [X, y] = tci.load_pickle(data_dir2, 'vst_transformed_data.pkl')

    # Return the data matrix (dataframe) and labels vector (series)
    return(X, y)


# Plot a PCA or tSNE analysis
def plot_unsupervised_analysis(results, y, figsize=(12,7.5), alpha=1, gray_indexes=None, ax=None, legend='full', plt_ctx=None):
    # Sample calls:
    #
    #   # Perform PCA
    #   import sklearn.decomposition as sk_decomp
    #   pca = sk_decomp.PCA(n_components=10)
    #   pca_res = pca.fit_transform(X.iloc[:,:500])
    #   log_fcn('Top {} PCA explained variance ratios: {}'.format(10, pca.explained_variance_ratio_))
    #   ax = tc.plot_unsupervised_analysis(pca_res, y)
    #   ax.set_title('PCA - variance-stabilizing transformation')
    #
    #   # Perform tSNE analysis
    #   import sklearn.manifold as sk_manif
    #   tsne = sk_manif.TSNE(n_components=2)
    #   tsne_res = tsne.fit_transform(X.iloc[:,:500])
    #   ax = tc.plot_unsupervised_analysis(tsne_res, y)
    #   ax.set_title('tSNE - variance-stabilizing transformation')
    #

    # Import relevant libraries
    import matplotlib.lines as mpl_lines
    import matplotlib.pyplot as plt
    import seaborn as sns

    import contextlib

    if not plt_ctx:
        plt_ctx = contextlib.nullcontext()

    # Get a reasonable set of markers and color palette
    markers = mpl_lines.Line2D.filled_markers
    nclasses = len(set(y))
    marker_list = markers * int(nclasses/len(markers)+1)
    color_palette = sns.color_palette("hls", nclasses)

    # Plot results
    with plt_ctx:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor='w')
        #ax = sns.scatterplot(x=results[:,0], y=results[:,1], hue=y, style=y, palette=color_palette, legend="full", alpha=alpha, markers=marker_list, edgecolor='k')
        #ax = sns.scatterplot(x=results[:,0], y=results[:,1], hue=y, style=y, palette=color_palette, legend=legend, alpha=(0.2 if gray_indexes is not None else alpha), markers=marker_list, edgecolor='k', ax=ax)

        ax = sns.scatterplot(x=results[:,0], y=results[:,1], hue=y, style=y, palette=color_palette, legend=legend, alpha=(0.2 if gray_indexes is not None else alpha), edgecolor='k', ax=ax)

        if gray_indexes is not None:
            import collections
            gray_indexes=list(collections.OrderedDict.fromkeys(gray_indexes.to_list()))
            #ax = sns.scatterplot(x=results[gray_indexes,0], y=results[gray_indexes,1], hue='gray', style=y.iloc[gray_indexes], palette=color_palette, markers=marker_list, edgecolor='k', ax=ax)
        # ax = sns.scatterplot(x=results[gray_indexes,0], y=results[gray_indexes,1], color='gray', style=y.iloc[gray_indexes], palette=color_palette, markers=marker_list, edgecolor='k', ax=ax, alpha=1, legend=legend)
            ax = sns.scatterplot(x=results[gray_indexes,0], y=results[gray_indexes,1], color='gray', style=y.iloc[gray_indexes], palette=color_palette, edgecolor='k', ax=ax, alpha=1, legend=legend)
        if legend is not False:
            ax.legend(bbox_to_anchor=(1,1))

        # if save_figure:
        #     fig.savefig(os.path.join(data_dir, 'pca_or_tsne_' + transformation_name_filename + '_transformation' + fn_addendum + '.png'), dpi=300, bbox_inches='tight')
        plt.show(fig)

    return(ax)


# Sample with replacement each label-group of the potentially unbalanced inputted data matrix and corresponding labels
# Return the corresponding balanced data matrix and corresponding labels, along with the numerical indexes that could be used to obtain these
def sample_populations(X2, y2, n=10, log_fcn=print):

    # Ensure the indexes of the input matrix and array match
    if not y2.index.equals(X2.index):
        log_fcn('ERROR: Indexes of input X and y do not match')
        exit()

    # Initialize the data matrix
    X = X2.copy()

    # Add the column of labels to the data matrix
    X['label'] = y2.copy()

    # Also add a column of the numerical indexes corresponding to the samples in X2 and y2
    X['index2'] = range(len(X))

    # Check that we did what we think we did
    if (not (X.iloc[:,-2] == y2).all()) or (not (X.iloc[:,:-2] == X2).all().all()):
        log_fcn('ERROR: We didn\'t correctly place the data and label matrices inside the combined data matrix')
        exit()

    # Sample with replacement each group (unique label) within the combined data matrix
    X = X.groupby('label').sample(n=n, replace=True)

    # Save the sampled labels and numerical indexes and drop those columns from the combined data matrix
    y = X['label']
    num_indexes = X['index2']
    X = X.drop(['label', 'index2'], axis='columns')

    # Ensure that all we really need is num_indexes
    if (not (y2.iloc[num_indexes] == y).all()) or (not (X2.iloc[num_indexes,:] == X).all().all()):
        log_fcn('ERROR: We cannot reproduce the results from num_indexes alone, as we should be able')
        exit()

    # Return the balanced data and labels and reproducing numerical indexes
    return(X, y, num_indexes)


# Create a figure helping to explore the extent of sampling each unique label in the dataset (i.e., each group)
def explore_sample_size(X, y, tsne_res, n_range=range(100,601,200), plt_ctx=None):

    # Import relevant library
    import matplotlib.pyplot as plt

    import contextlib

    if not plt_ctx:
        plt_ctx = contextlib.nullcontext()

    # Constant
    base_figsize = (16,5) # this is the size of an entire two-image row

    # Get the list of possible values of n from its inputted range
    n_values = [x for x in n_range]
    nn = len(n_values)

    # Initialize the figure
    with plt_ctx:
        fig, axs = plt.subplots(nrows=nn, ncols=2, figsize=(base_figsize[0], base_figsize[1]*nn), squeeze=False, facecolor='w')

        # For each sampling size...
        for n_ind, n in enumerate(n_values):

            # Sample the imbalanced dataset
            _, _, num_indexes = sample_populations(X, y, n=n)

            # Plot the sampling results themselves
            ax = plot_unsupervised_analysis(tsne_res[num_indexes,:], y.iloc[num_indexes], alpha=0.5, ax=axs[n_ind,0], legend=False) # note y = y2.iloc[num_indexes]
            ax.set_title('tSNE - VST - n={} - sample'.format(n))

            # Now plot the sampling results on top of the tSNE of the full dataset in order to see how much we covered
            ax = plot_unsupervised_analysis(tsne_res, y, gray_indexes=num_indexes, ax=axs[n_ind,1])
            ax.set_title('tSNE - VST - n={} - sample in gray on top of original'.format(n))
        plt.show(fig)


    return(fig)


# Run some random forest classification models on the data, saving the results and calculating the accuracies of the models on the entire input dataset (i.e., our test set, which includes the training data, which is obtained by bootstrap sampling within the classes)
# This bootstrap sampling from each class is sort of what we're forced to do given the small size of the minority classes, though we see the accuracy is still so good that we can probably do a "real" study (i.e., with a training and test set)
#def calculate_whole_dataset_accuracy_vs_bootstrap_sampling_size(X, y, project_directory, possible_n=None, ntrials=10):
def generate_random_forest_models(X, y, project_directory, study_name, possible_n=None, ntrials=10, log_fcn=print):

    # Import relevant modules
    import os

    import numpy as np
    import sklearn.ensemble as sk_ens
    tci = get_tci_library()

    # Constant
    datadir = os.path.join(project_directory, 'data')

    # Set the possible sampling sizes to a reasonable default if it's not specified in the function call
    if possible_n is None:
        possible_n = [x for x in range(1,13)] + [x for x in range(15,46,5)] + [x for x in range(50,101,10)]
    n_sample_sizes = len(possible_n)

    # Initialize the accuracy-holder array
    accuracies = np.zeros((ntrials, n_sample_sizes))
    rnd_clf_holder = []

    # If the datafile does not already exist...
    if not os.path.exists(os.path.join(datadir, study_name+'.pkl')):

        # For each sampling size, for each trial...
        for itrial in range(ntrials):
            log_fcn('On trial {} of {}...'.format(itrial+1, ntrials))

            rnd_clf_holder_inside = []
            for iin, n in enumerate(possible_n):
                log_fcn('  On sample size {} of {} (n={})...'.format(iin+1, n_sample_sizes, n))

                # Sample the input dataset using the current sampling size n
                X_bal, y_bal, _ = sample_populations(X, y, n=n)

                # Check feature equality
                if not X.columns.equals(X_bal.columns):
                    log_fcn('ERROR: Imbalanced and balanced features are not the same')
                    exit()

                # Fit a random forest classifier to the sampled, balanced dataset
                clf = sk_ens.RandomForestClassifier()
                clf.fit(X_bal, y_bal)

                # Test the fitted model on the full, input dataset
                accuracies[itrial, iin] = clf.score(X, y)

                # Save all other data
                #rnd_clf_holder_inside.append([itrial, iin, n, X_bal, y_bal, clf.feature_importances_, clf.predict_proba(X), clf.n_features_])
                rnd_clf_holder_inside.append([itrial, iin, n, y_bal, clf])

            rnd_clf_holder.append(rnd_clf_holder_inside)

        # Save the data to disk
        tci.make_pickle([accuracies, possible_n, rnd_clf_holder], datadir, study_name+'.pkl')

    # Otherwise, read it in
    else:
        [accuracies, possible_n, rnd_clf_holder] = tci.load_pickle(datadir, study_name+'.pkl')

    return(accuracies, possible_n, rnd_clf_holder)


# Plot the mean accuracy vs. the sampling sizes with error bars indicating the minimum and maximum accuracies over all the trials
def plot_accuracy_vs_sample_size(accuracies, possible_n, study_name, plt_ctx=None):

    # Import relevant libraries

    import matplotlib.pyplot as plt
    import numpy as np

    import contextlib

    if not plt_ctx:
        plt_ctx = contextlib.nullcontext()

    # Calculate the average accuracy over all the trials
    means = accuracies.mean(axis=0)

    # Plot the mean accuracy vs. the sampling sizes with error bars indicating the minimum and maximum accuracies over all the trials
    with plt_ctx:
        fig, ax = plt.subplots(figsize=(10,6), facecolor='w')
        _ = ax.errorbar(x=possible_n, y=means, yerr=np.row_stack((means-accuracies.min(axis=0), accuracies.max(axis=0)-means)), fmt='*-', ecolor='red', capsize=4)
        ax.grid(True)
        ax.set_xlabel('Sample size for each class')
        ax.set_ylabel('Accuracy')
        ax.set_title('Overall accuracy on input dataset - ' + study_name)
        plt.show(fig)


# Determine the genes in decreasing order of average feature importance, calculating all necessary importance metrics along the way
def calculate_average_feature_importance(feature_names, rnd_clf_holder, num_last_sample_sizes=10, do_printing=False, log_fcn=print):

    # Sample call: calculate_average_feature_importance(X2.columns, rnd_clf_holder, num_last_sample_sizes=10)

    # Import relevant libraries
    import numpy as np
    import pandas as pd

    # Constant
    #tol = 1e-8
    tol = 1e-16

    # Get the big numpy array of feature importances
    n_features = len(feature_names)
    importances = np.zeros((n_features, len(rnd_clf_holder), len(rnd_clf_holder[0])))
    for itrial, rnd_clf_holder_inside in enumerate(rnd_clf_holder): # ntrials of these
        for isample_size, model_data in enumerate(rnd_clf_holder_inside): # len(possible_n) of these; model_data is [itrial, iin, n, y_bal, clf]
            importances[:,itrial,isample_size] = model_data[4].feature_importances_

    # Save the full importances holder
    #importances_full = importances.copy()

    # Average over the trials and the last num_last_sample_sizes sample sizes
    importances = importances[:,:,-num_last_sample_sizes:].mean(axis=(1,2))

    # Sort the importances in decreasing order
    order = (-importances).argsort(axis=0)

    # Initialize the arrays that will go into the final dataframe containing the importance information
    x = importances[order]
    x_copy = x.copy()
    place_arr = np.zeros((n_features,), dtype=object)
    norm_score_arr = np.zeros((n_features,), dtype=float)
    num_in_rank_arr = np.zeros((n_features,), dtype=int)
    raw_score_arr = np.zeros((n_features,), dtype=float)

    # Keep just the importance scores that are not zero
    x = x[np.abs(x)>=tol]

    # Obtain the indexes of where the feature score changes
    diffs = x[1:] - x[:-1]
    rank_changed_loc = np.append(np.where((np.abs(diffs)>=tol))[0], len(x)-1)

    # For each location at which the score changes...
    start_ind = 0
    place = 1
    for loc in rank_changed_loc:

        # Determine the last range index for the current set of continuous scores
        stop_ind = loc + 1

        # Count the number of features having the current score
        num_in_rank = stop_ind - start_ind

        # Get the current score for the set of features having this score (I don't have to do it this way, but it is technically the fairest way to do it)
        raw_score = x[start_ind:stop_ind].mean()

        # Normalize this score so that the largest score for this dataset is 1 (and the smallest is zero)
        norm_score = raw_score / x[0] # can divide by the first element since they're ordered in decreasing order and hence the first element is the largest

        # Prepend 't-' to the rank if there are multiple features having this score in order to indicate a 'tie'
        tie_string = ('t-' if num_in_rank>1 else '')

        # Set the corresponding elements in the arrays
        if do_printing:
            log_fcn('place={},\tnorm_score={:4.2f}, num_in_rank={}, raw_score={:7.5f}, start_ind={}, stop_ind={}'.format(tie_string+str(place), norm_score, num_in_rank, raw_score, start_ind, stop_ind))
        place_arr[start_ind:stop_ind] = tie_string + str(place)
        norm_score_arr[start_ind:stop_ind] = norm_score
        num_in_rank_arr[start_ind:stop_ind] = num_in_rank
        raw_score_arr[start_ind:stop_ind] = raw_score

        # Update the starting index and the place
        start_ind = stop_ind
        place = place + 1

    # Create a Pandas dataframe including the average importances (in "original score") with the corresponding gene names as indexes in descending order, removing the version numbers from the Ensembl gene IDs
    #important_genes = pd.Series(data=importances[order], index=[x.split('.')[0] for x in feature_names[order]], name='score')
    important_genes = pd.DataFrame(data={'place': place_arr, 'norm_score': norm_score_arr, 'num_in_rank': num_in_rank_arr, 'raw_score': raw_score_arr, 'original score': x_copy}, index=[x.split('.')[0] for x in feature_names[order]])

    #return(important_genes, importances_full)
    return(important_genes)
