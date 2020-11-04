# Instantiation of this class mainly loads Consolidata_data.txt into a Pandas dataframe (or reads in a simulated one in the case of simulated data) and performs some preprocessing on it
# It will create a pickle file (initial_data.pkl) of the read-in and preprocessed data, unless the file already exists, in which case this step is skipped
class TIMECellInteraction:

    def __init__(self, pickle_dir, simulate_data, allow_compound_species, mapping_dict, **kwargs):

        # Import relevant module
        import os

        # These next two blocks aren't technically needed but it helps to set these here to help for linting purposes
        
        # These are set in this method but not saved in the traditional way (instead, using make_pickle_dict())
        self.pickle_dir = None
        self.unique_species = []
        self.doubling_type = None
        self.unique_slides = []

        self.is_real_data = None
        self.compound_species_allowed = None
        self.csv_file = None
        self.plotting_map = []
        self.num_colors = None

        # These are set in other functions in this class but not saved in the traditional way (instead, using make_pickle_dict())
        self.data_by_slide = []
        self.dr = None
        self.k_max = None
        self.min_nvalid_centers = None

        # Assign local variables that aren't the same as those inputted in order to save them later using make_pickle_dict()
        is_real_data = not simulate_data
        compound_species_allowed = allow_compound_species

        # Constant
        pickle_file = 'initial_data.pkl'

        # If the pickle file doesn't exist...
        if not os.path.exists(os.path.join(pickle_dir, pickle_file)):

            # If requesting simulated data...
            if simulate_data:

                # ...generate simulated data
                midpoints = kwargs['midpoints']
                max_real_area = kwargs['max_real_area']
                coord_spacing = kwargs['coord_spacing']
                mult = kwargs['mult']
                self.doubling_type = kwargs['doubling_type']
                csv_file = None
                self.data = get_simulated_data(kwargs['doubling_type'], midpoints, max_real_area, coord_spacing, mult)

            else:

                # ...otherwise, read in the data from the CSV file
                self.csv_file = kwargs['csv_file']
                doubling_type = None
                self.data = get_consolidated_data(kwargs['csv_file'])

            # Preprocess the data, i.e., the pandas dataframe
            self.phenotypes = self.preprocess_dataframe(allow_compound_species)

            # Get the plotting map, number of colors, unique species, and the unique slides
            plotting_map, num_colors, unique_species, unique_slides = get_dataframe_info(self.data, self.phenotypes, mapping_dict)

            # Save the data to a pickle file
            self.make_pickle_dict(['pickle_dir', 'is_real_data', 'compound_species_allowed', 'doubling_type', 'csv_file', 'data', 'phenotypes', 'plotting_map', 'num_colors', 'unique_species', 'unique_slides'], locals(), pickle_file)

        else:

            # Load the data from the pickle file if it already exists
            self.load_pickle_dict(pickle_file, pickle_dir=pickle_dir)


    # For all slides/ROIs, store the largest number of neighbors and the largest population density in any slice and for any species of center-neighbor pair
    # Calculate a nice, consistent, global set of edges/midpoints for the corresponding maximum numbers of neighbors and population density
    def calculate_bins(self):

        # Import relevant module
        import numpy as np

        # Set variables already defined as attributes
        data_by_slide = self.data_by_slide
        dr = self.dr
        k_max = self.k_max

        # Variables
        area = np.pi*dr*2*dr*(np.arange(k_max)+1) - np.pi*dr*dr
        nneighbors_max = 0
        pop_dens_max = 0

        # For every slide...
        for data_by_roi in data_by_slide:
            [_, _, roi_data] = data_by_roi

            # For every ROI in the slide...
            for roi_data_item in roi_data:
                [_, _, _, _, _, _, _, _, _, _, _, valid_centers, nneighbors, _, _, nvalid_species, _] = roi_data_item

                # For every center species...
                for icenter_spec in range(nvalid_species):

                    # For every neighbor species...
                    for ineighbor_spec in range(nvalid_species):

                        # For every slice...
                        for ik in range(k_max):

                            # Calculate the actual number of neighbors around the centers
                            valid_centers3 = valid_centers[:,ik,icenter_spec] # centers of species icenter_spec that are valid at radius ik
                            to_histogram = nneighbors[valid_centers3,ik,ineighbor_spec] # before we replaced nneighbors2 in this line with pop_dens2

                            # Get the largest number of neighbors and set to the global maximum if it's larger than the current global maximum
                            curr_max = np.max(to_histogram)
                            if curr_max > nneighbors_max:
                                nneighbors_max = curr_max

                            # Get the largest population density and set to the global maximum if it's larger than the current global maximum
                            curr_max2 = curr_max / area[ik]
                            if curr_max2 > pop_dens_max:
                                pop_dens_max = curr_max2

        # Define the just-large-enough bins and edges
        midpoints = np.arange(nneighbors_max+1)
        edges = np.arange(nneighbors_max+2) - 0.5
        edges_pop_dens = np.linspace(0,pop_dens_max,11)
        midpoints_pop_dens = (edges_pop_dens[1]-edges_pop_dens[0])/2 + edges_pop_dens[0:-1]

        # Set calculated attributes
        self.midpoints = midpoints
        self.edges = edges
        self.midpoints_pop_dens = midpoints_pop_dens
        self.edges_pop_dens = edges_pop_dens


    # Calculate the densities for every slide/ROI for every center-neighbor species pair (but we're summing over radius), two different ways as a sanity check
    # So in the end we end up with the average number of neighbors within the outer radius. Note that despite this we are using all possible data, i.e., more data for smaller radii
    def calculate_densities(self):

        # Import relevant module
        import numpy as np

        # Set variables already defined as attributes
        data_by_slide = self.data_by_slide
        pdfs = self.pdfs
        midpoints = self.midpoints
        k_max = self.k_max
        dr = self.dr

        # Constant
        tol = 1e-7
        
        # Variable
        nbins = len(midpoints)

        # Area consistency check
        area = np.pi*dr*2*dr*(np.arange(k_max)+1) - np.pi*dr*dr
        if np.abs(np.sum(area, axis=0)-(np.pi*(dr*k_max)**2)) > tol:
            print('ERROR: Area check failed')
            exit()

        # For every slide...
        density_slide = []
        for islide, data_by_roi in enumerate(data_by_slide):
            print('On slide {} of {}...'.format(islide+1, len(data_by_slide)))
            [_, _, roi_data] = data_by_roi

            # For every ROI in the slide...
            density_roi = []
            for iroi, roi_data_item in enumerate(roi_data):
                print('  On ROI {} of {}...'.format(iroi+1, len(roi_data)))
                [_, _, _, _, _, _, _, _, _, _, _, valid_centers, nneighbors, _, _, nvalid_species, _] = roi_data_item

                # Calculate the density using the PDF
                curr_pdfs = pdfs[islide][iroi] # pdfs: (nbins, k_max, nvalid_species, nvalid_species)
                midpoints_tiled = np.tile(np.reshape(midpoints,(nbins,1,1,1)), (1,k_max,nvalid_species,nvalid_species))
                density_pdf = np.sum(midpoints_tiled * curr_pdfs, axis=(0,1))
                            
                # For every center species...
                density_raw = np.zeros((nvalid_species,nvalid_species))
                for icenter_spec in range(nvalid_species):

                    # For every neighbor species...
                    for ineighbor_spec in range(nvalid_species):

                        # Calculate the density using the raw counts of the numbers of neighbors
                        mean_holder = np.zeros((k_max,))
                        for ik in range(k_max):
                            mean_holder[ik] = np.mean(nneighbors[valid_centers[:,ik,icenter_spec],ik,ineighbor_spec], axis=0)
                        density_raw[icenter_spec,ineighbor_spec] = np.sum(mean_holder, axis=0)

                        # Check that the two density calculation methods agree... note the PDF method should be wrong if the max bin size isn't large enough; this died for the simulated data but passed for the real data which shows that the bins I'm using for all real data are sufficient
                        if np.abs(density_raw[icenter_spec,ineighbor_spec] - density_pdf[icenter_spec,ineighbor_spec]) > tol:
                            print('ERROR: Sanity check failed')
                            print((density_raw[icenter_spec,ineighbor_spec], density_pdf[icenter_spec,ineighbor_spec]))
                            for ik in range(k_max):
                                print(np.max(nneighbors[valid_centers[:,ik,icenter_spec],ik,ineighbor_spec]))
                            exit()

        # Set the calculated attribute, density_slide
                density_roi.append(density_pdf)
            density_slide.append(density_roi)
        self.density_tot = density_slide


    # Create one pickle file (null_data-slide_...) per slide, if it doesn't already exist, keeping the frequencies same for every center and neighbor species in each ROI, containing the numbers of neighbors counts for each of nsamplings samplings of both random and Keren-like distributions (null distributions) in order to compare our actual data to
    # Note that upon performing the neighbor counting, we're reducing min_nvalid_centers by a factor of 10 since there will generally be fewer than normal valid centers
    def calculate_null_distributions(self, nsamplings=500):

        # Import relevant modules
        import numpy as np
        import os

        # Set variables already defined as attributes
        k_max = self.k_max
        data_by_slide = self.data_by_slide
        min_nvalid_centers = self.min_nvalid_centers
        coord_spacing = self.coord_spacing
        pickle_dir = self.pickle_dir

        # For every slide...
        null_data_by_slide = []
        for islide, data_by_roi in enumerate(data_by_slide):
            print('On slide {} of {}...'.format(islide+1, len(data_by_slide)))

            # Define the pickle file for the current slide
            pickle_file = 'null_data-slide_{:03d}-nsamplings_{}.pkl'.format(islide+1, nsamplings)

            # If the pickle file doesn't already exist...
            if not os.path.exists(os.path.join(pickle_dir,pickle_file)):
                [_, _, roi_data] = data_by_roi

                # For every ROI in the slide...
                null_data_by_roi = []
                for iroi, roi_data_item in enumerate(roi_data):
                    print('  On ROI {} of {}...'.format(iroi+1, len(roi_data)))
                    [x_roi, y_roi, species_roi, _, _, _, _, _, _, _, _, _, _, _, valid_species, nvalid_species, _] = roi_data_item

                    # Get all possible coordinate values in the current ROI
                    npossible_x = int(((x_roi.max()-x_roi.min())/coord_spacing+1))
                    npossible_y = int(((y_roi.max()-y_roi.min())/coord_spacing+1))
                    possible_x = np.linspace(x_roi.min(), x_roi.max(), num=npossible_x)
                    possible_y = np.linspace(y_roi.min(), y_roi.max(), num=npossible_y)

                    # For every center species...
                    null_data_by_center = []
                    for icenter_spec in range(nvalid_species):
                        print('    On center {} of {}...'.format(icenter_spec+1, nvalid_species))
                        center_spec = valid_species[icenter_spec]

                        # For every neighbor species...
                        null_data_by_neighbor = []
                        for ineighbor_spec in range(nvalid_species):
                            print('      On neighbor {} of {}...'.format(ineighbor_spec+1, nvalid_species))
                            neighbor_spec = valid_species[ineighbor_spec]

                            # For the current center-neighbor pair, get the corresponding indexes, as well as the footprint indexes
                            center_ind = np.nonzero(species_roi == center_spec)[0]
                            neighbor_ind = np.nonzero(species_roi == neighbor_spec)[0]
                            footprint_ind = np.nonzero(species_roi != center_spec)[0]

                            # Get the numbers of centers, neighbors, and footprints
                            ncenters = len(center_ind)
                            nneighbors = len(neighbor_ind)
                            nfootprints = len(footprint_ind)

                            # Get arrays of the center and neighbor species IDs
                            center_spec_arr = np.ones((ncenters,), dtype='uint64')*center_spec
                            neighbor_spec_arr = np.ones((nneighbors,), dtype='uint64')*neighbor_spec

                            # Get completely random coordinates in the ROI of the centers and neighbors, keeping their frequencies the same as they currently are
                            # For the neighbors, get such a random set of coordinates for each bootstrap sample
                            center_x_rand = possible_x[np.random.randint(npossible_x, size=(ncenters,))]
                            center_y_rand = possible_y[np.random.randint(npossible_y, size=(ncenters,))]
                            neighbor_x_rand = possible_x[np.random.randint(npossible_x, size=(nneighbors, nsamplings))]
                            neighbor_y_rand = possible_y[np.random.randint(npossible_y, size=(nneighbors, nsamplings))]

                            # Get the actual center coordinates
                            center_x_orig = x_roi[center_ind]
                            center_y_orig = y_roi[center_ind]

                            # Get nneighbors random footprints, one for each bootstrap sample
                            if nfootprints > 0:
                                neighbor_x_foot = x_roi[footprint_ind[np.random.randint(nfootprints, size=(nneighbors, nsamplings))]]
                                neighbor_y_foot = y_roi[footprint_ind[np.random.randint(nfootprints, size=(nneighbors, nsamplings))]]

                            # Store the types of center coordinates to use for the null distributions, one set of coordinates for ALL bootstrap samples (ncenters,)
                            center_x_all = (center_x_rand, center_x_orig)
                            center_y_all = (center_y_rand, center_y_orig)
                            ncenter_all = len(center_x_all)

                            # Store the types of neighbor coordinates to use for the null distributions, one set of coordinates for EACH bootstrap sample (nneighbors, nsamplings)
                            if nfootprints > 0:
                                neighbor_x_all = (neighbor_x_rand, neighbor_x_foot)
                                neighbor_y_all = (neighbor_y_rand, neighbor_y_foot)
                            else:
                                neighbor_x_all = [neighbor_x_rand]
                                neighbor_y_all = [neighbor_y_rand]
                            nneighbor_all = len(neighbor_x_all)

                            # Determine whether the current center and neighbor species are the same
                            single_species = center_spec == neighbor_spec

                            # For the current null ROI, determine the total number of cells, number of unique species, and species ID array
                            if single_species:
                                ncells = nneighbors
                                nunique_species = 1
                                spec_arr = neighbor_spec_arr
                            else:
                                ncells = ncenters + nneighbors
                                nunique_species = 2
                                spec_arr = np.r_[center_spec_arr, neighbor_spec_arr]

                            # Initialize arrays to save below
                            nneighbors_null = np.zeros((ncells, k_max, nunique_species, nsamplings, ncenter_all, nneighbor_all), dtype='uint16')
                            valid_centers_null = np.zeros((ncells, k_max, nunique_species, nsamplings, ncenter_all, nneighbor_all), dtype='bool')
                            returned_species_null = np.zeros((nunique_species, nsamplings, ncenter_all, nneighbor_all), dtype='uint32') # the IDs of the valid unique species
                            last_nvalid_centers_null = np.zeros((nunique_species, nsamplings, ncenter_all, nneighbor_all), dtype='uint32') # counts of the number of valid centers for just the valid (minimally numbering) unique species
                            
                            # For every bootstrap sample...
                            for isampling in range(nsamplings):

                                # For each type of center coordinates...                                
                                for icenter, (curr_center_x, curr_center_y) in enumerate(zip(center_x_all, center_y_all)):

                                    # For each type of neighbor coordinates...
                                    for ineighbor, (curr_neighbor_x, curr_neighbor_y) in enumerate(zip(neighbor_x_all, neighbor_y_all)):

                                        # Get the current set of coordinates on which to perform the neighbor counting
                                        if single_species:
                                            x = curr_neighbor_x[:,isampling]
                                            y = curr_neighbor_y[:,isampling]
                                        else:
                                            x = np.r_[curr_center_x, curr_neighbor_x[:,isampling]]
                                            y = np.r_[curr_center_y, curr_neighbor_y[:,isampling]]

                                        # Perform the neighbor counting, reducing min_nvalid_centers since there will generally be fewer than normal valid centers
                                        _, _, _, _, _, _, _, _, _, tmp_valid_centers_null, tmp_nneighbors_null, _, _, tmp_returned_species_null, nvalid_species_null, tmp_last_nvalid_centers_null, _, _, _, _ = self.count_neighbors_in_roi_class(x, y, spec_arr, do_printing=False, min_nvalid_centers=int(np.round(min_nvalid_centers/10)))

                                        # Populate the arrays of interest
                                        valid_centers_null[:,:,:,isampling,icenter,ineighbor] = tmp_valid_centers_null
                                        nneighbors_null[:,:,:,isampling,icenter,ineighbor] = tmp_nneighbors_null
                                        returned_species_null[:,isampling,icenter,ineighbor] = tmp_returned_species_null
                                        last_nvalid_centers_null[:,isampling,icenter,ineighbor] = tmp_last_nvalid_centers_null
                                                
                                        # Ensure that the expected number of species was returned and that broadcasting was not performed (which indicates that something unexpected occurred)
                                        # nvalid_species_null: number of valid unique species
                                        if (nvalid_species_null != nunique_species) or (valid_centers_null[:,:,:,isampling,icenter,ineighbor].shape != tmp_valid_centers_null.shape) or (nneighbors_null[:,:,:,isampling,icenter,ineighbor].shape != tmp_nneighbors_null.shape) or (returned_species_null[:,isampling,icenter,ineighbor].shape != tmp_returned_species_null.shape) or (last_nvalid_centers_null[:,isampling,icenter,ineighbor].shape != tmp_last_nvalid_centers_null.shape):
                                            print('ERROR: Something improbable happened')
                                            print((nvalid_species_null != nunique_species), (valid_centers_null[:,:,:,isampling,icenter,ineighbor].shape != tmp_valid_centers_null.shape), (nneighbors_null[:,:,:,isampling,icenter,ineighbor].shape != tmp_nneighbors_null.shape), (returned_species_null[:,isampling,icenter,ineighbor].shape != tmp_returned_species_null.shape), (last_nvalid_centers_null[:,isampling,icenter,ineighbor].shape != tmp_last_nvalid_centers_null.shape))
                                            exit()

                            # Store all data into a tuple
                            null_data_by_neighbor.append(((center_spec, neighbor_spec), (ncenters, nneighbors, nfootprints), ncells, nunique_species, nsamplings, center_x_all, center_y_all, neighbor_x_all, neighbor_y_all, spec_arr, valid_centers_null, nneighbors_null, returned_species_null, last_nvalid_centers_null))

                # Save the data for every slide
                        null_data_by_center.append(null_data_by_neighbor)
                    null_data_by_roi.append(null_data_by_center)
                make_pickle(null_data_by_roi, pickle_dir, pickle_file)

            # If the pickle file already exists, just read it in
            else:
                null_data_by_roi = self.load_pickle_class(pickle_file)

            # Store the data for all slides into null_data_by_slide
            null_data_by_slide.append(null_data_by_roi)

        # Set the "calculated" attributes
        self.null_data_by_slide = null_data_by_slide
        self.nsamplings = nsamplings


    # Save to (or load from) null_properties.pkl the means and standard deviations of the null distributions (for both the simulated null data and the theoretical distribution)
    # First, calculate the PDFs for all the null distributions (including the theoretical one)
    def calculate_null_properties(self):
        
        # Null distribution methods:
        #
        #   nneighbor_all==2:
        #
        #   0: theoretical Poisson
        #   1: center=rand, neighbor=rand
        #   2: center=rand, neighbor=foot
        #   3: center=orig, neighbor=rand
        #   4: center=orig, neighbor=foot
        #
        #   nneighbor_all==1:
        #
        #   0: theoretical Poisson
        #   1: center=rand, neighbor=rand
        #   2: center=orig, neighbor=rand # changed from 3 above to 2 here
        #
        #   So if nneighbor_all==1, then loop method index (imethod0 below) of 2 should go to general method index of 3

        # Import relevant modules
        import numpy as np
        import os
        
        # Set variables already defined as attributes
        null_data_by_slide = self.null_data_by_slide
        k_max = self.k_max
        nsamplings = self.nsamplings
        data_by_slide = self.data_by_slide
        dr = self.dr
        pickle_dir = self.pickle_dir

        # Constants; see comments at top of this function
        nmethods = 5
        bad_val = -9999
        pickle_file = 'null_properties.pkl'
        tol = 1e-8

        # Variable
        areas = np.pi*dr*2*dr*(np.arange(k_max)+1) - np.pi*dr*dr

        # If the pickle file doesn't exist...
        if not os.path.exists(os.path.join(pickle_dir, pickle_file)):

            # For every slide...
            null_properties_by_slide = []
            for islide, (null_data_by_roi, data_by_roi) in enumerate(zip(null_data_by_slide, data_by_slide)):
                print('On slide {} of {}...'.format(islide+1, len(null_data_by_slide)))
                [_, _, roi_data] = data_by_roi

                # For every ROI...
                null_properties_by_roi = []
                for iroi, (null_data_by_center, roi_data_item) in enumerate(zip(null_data_by_roi, roi_data)):
                    print('  On ROI {} of {}...'.format(iroi+1, len(null_data_by_roi)))
                    [x_roi, y_roi, species_roi, roi_size, _, _, _, _, _, _, _, valid_centers_exper, nneighbors_exper, _, valid_species, nvalid_species, _] = roi_data_item

                    # Calculate edges and nbins (technically just for the null distribution-based data, but we're using the resulting values of midpoints, edges, and nbins also for the theoretical data so that we can put all null data into the same arrays, e.g., the pdfs array)
                    # See rest of function for comments, as this block is similar to the rest of the function
                    nneighbors_max = 0
                    for null_data_by_neighbor in null_data_by_center:
                        for curr_null_data in null_data_by_neighbor:
                            (center_spec, neighbor_spec), _, _, _, nsamplings, center_x_all, _, neighbor_x_all, _, _, valid_centers_null, nneighbors_null, returned_species_null, _ = curr_null_data
                            ncenter_all = len(center_x_all)
                            nneighbor_all = len(neighbor_x_all)
                            for isampling in range(nsamplings):
                                for icenter in range(ncenter_all):
                                    for ineighbor in range(nneighbor_all):
                                        curr_nneighbors = nneighbors_null[:,:,:,isampling,icenter,ineighbor]
                                        curr_valid_centers = valid_centers_null[:,:,:,isampling,icenter,ineighbor]
                                        curr_returned_species = returned_species_null[:,isampling,icenter,ineighbor]
                                        icenter_spec2 = np.nonzero(curr_returned_species==center_spec)[0]
                                        ineighbor_spec2 = np.nonzero(curr_returned_species==neighbor_spec)[0]
                                        for ik in range(k_max):
                                            valid_centers3 = curr_valid_centers[:,ik,icenter_spec2].flatten()
                                            to_histogram = curr_nneighbors[valid_centers3,ik,ineighbor_spec2]
                                            to_histogram_max = to_histogram.max(axis=0)
                                            if to_histogram_max > nneighbors_max:
                                                nneighbors_max = to_histogram_max
                    midpoints = np.arange(nneighbors_max+1)
                    edges = np.arange(nneighbors_max+2) - 0.5
                    nbins = nneighbors_max + 1


                    #### Null distribution quantity (pdfs_roi) calculations #############################
                    # Initialize the main array of interest
                    pdfs_roi = np.ones((nbins, k_max, nvalid_species, nvalid_species, nmethods, nsamplings))*bad_val

                    # For every center-neighbor pair...
                    for icenter_spec, null_data_by_neighbor in enumerate(null_data_by_center):
                        for ineighbor_spec, curr_null_data in enumerate(null_data_by_neighbor):

                            # Extract the required data
                            (center_spec, neighbor_spec), _, _, _, nsamplings, center_x_all, _, neighbor_x_all, _, _, valid_centers_null, nneighbors_null, returned_species_null, _ = curr_null_data
                            ncenter_all = len(center_x_all)
                            nneighbor_all = len(neighbor_x_all)

                            # For each bootstrap sample...
                            for isampling in range(nsamplings):

                                # For each neighbor/center "method"...
                                imethod0 = 0
                                for icenter in range(ncenter_all):
                                    for ineighbor in range(nneighbor_all):
                                        imethod0 = imethod0 + 1
                                        imethod = imethod0
                                        if (nneighbor_all==1) and (imethod0==2):
                                            imethod = 3

                                        # Get the data for the current bootstrap sample and method
                                        curr_nneighbors = nneighbors_null[:,:,:,isampling,icenter,ineighbor]
                                        curr_valid_centers = valid_centers_null[:,:,:,isampling,icenter,ineighbor]
                                        curr_returned_species = returned_species_null[:,isampling,icenter,ineighbor] # (nunique_species, nsamplings, ncenter_all, nneighbor_all)

                                        # Get the indexes corresponding to the current center-neighbor pair in the correct order
                                        icenter_spec2 = np.nonzero(curr_returned_species==center_spec)[0]
                                        ineighbor_spec2 = np.nonzero(curr_returned_species==neighbor_spec)[0]

                                        # For each radial slice, calculate the PDFs
                                        for ik in range(k_max):

                                            # Calculate the pdfs
                                            valid_centers3 = curr_valid_centers[:,ik,icenter_spec2].flatten()
                                            to_histogram = curr_nneighbors[valid_centers3,ik,ineighbor_spec2]
                                            pdfs_roi[:,ik,icenter_spec,ineighbor_spec,imethod,isampling] = np.histogram(to_histogram, bins=edges, density=True)[0]
                    #####################################################################################


                    #### Theoretical distribution quantity calculations #################################
                    # Here we are basically calculating the empirical PDFs but based on the theoretical Poisson function as opposed to actual data

                    # Initialize "helper values" (and their checks) needed for the formulas
                    nvalid_holder = np.ones((k_max, nvalid_species))*bad_val # numbers of valid centers
                    nvalid_holder_check = np.ones((k_max, nvalid_species, nvalid_species))*bad_val
                    nexpected_holder = np.ones((k_max, nvalid_species))*bad_val # number of expected neighbors in each slice
                    nexpected_holder_check = np.ones((k_max, nvalid_species, nvalid_species))*bad_val
                    pois_holder = np.ones((nbins, k_max, nvalid_species))*bad_val # relative expected number of neighbors in bin b and slice k given the neighbor density in the ROI
                    pois_holder_check = np.ones((nbins, k_max, nvalid_species, nvalid_species))*bad_val

                    # For every center species...
                    neighbor_freq_in_roi = np.zeros((nvalid_species,))
                    center_coords_holder = []
                    for icenter_spec in range(nvalid_species):
                        center_spec = valid_species[icenter_spec]

                        # Define some variables
                        spec_loc_bool = species_roi==center_spec
                        neighbor_freq_in_roi[icenter_spec] = (spec_loc_bool).sum(axis=0)
                        x_spec = x_roi[spec_loc_bool]
                        y_spec = y_roi[spec_loc_bool]
                        center_coords_holder.append(np.c_[x_spec,y_spec])

                        # For every neighbor species...
                        for ineighbor_spec in range(nvalid_species):
                            neighbor_spec = valid_species[ineighbor_spec]

                            # For every radial slice...
                            for ik in range(k_max):

                                # Calculate the actual number of neighbors around the centers
                                valid_centers3_exper = valid_centers_exper[:,ik,icenter_spec] # centers of species icenter_spec that are valid at radius ik
                                to_histogram = nneighbors_exper[valid_centers3_exper,ik,ineighbor_spec] # before we replaced nneighbors2 in this line with pop_dens2

                                # Calculate the "helper values"
                                nvalid_tmp = len(to_histogram)
                                nexpected_tmp = (species_roi==neighbor_spec).sum(axis=0) / np.array(roi_size).prod() * areas[ik]
                                pois_tmp = poisson(nexpected_tmp, midpoints) #### FIX ME!!!! #### Actually I think this is right!! (I don't think you should normalize a theoretical distribution, see my notebook notes on 5/6/20)
                                # tmp = poisson(nexpected_tmp, midpoints)
                                # pois_tmp = tmp / tmp.sum(axis=0)

                                # Store the helper values
                                nvalid_holder[ik,icenter_spec] = nvalid_tmp
                                nexpected_holder[ik,ineighbor_spec] = nexpected_tmp
                                pois_holder[:,ik,ineighbor_spec] = pois_tmp

                                # Run some checks
                                if nvalid_tmp != valid_centers3_exper.sum(axis=0):
                                    print('ERROR: Inconsistent number of valid centers')
                                    exit()
                                nvalid_holder_check[ik,icenter_spec,ineighbor_spec] = nvalid_tmp
                                nexpected_holder_check[ik,icenter_spec,ineighbor_spec] = nexpected_tmp
                                pois_holder_check[:,ik,icenter_spec,ineighbor_spec] = pois_tmp

                    # Check whether the coordinates are the same
                    all_coords_are_same = False
                    if neighbor_freq_in_roi.var(axis=0) < tol:
                        center_coords_holder2 = np.zeros((int(neighbor_freq_in_roi[0]),2,nvalid_species))
                        for ispec, curr_coords in enumerate(center_coords_holder):
                            center_coords_holder2[:,:,ispec] = curr_coords
                        if center_coords_holder2.var(axis=2).max() < tol:
                            all_coords_are_same = True
                    #####################################################################################


                    #### Calculate the null distribution-based means and stds ###########################
                    # Assign the values needed for the fomulas to short-named variables, similar to the names in the formulas
                    pdf_bk = pdfs_roi                                                        # (nbins, k_max, nvalid_species, nvalid_species, nmethods, nsamplings)
                    dens_k = ( midpoints.reshape((nbins,1,1,1,1,1)) * pdfs_roi ).sum(axis=0) # (       k_max, nvalid_species, nvalid_species, nmethods, nsamplings)

                    # Calculate the means and stds for the densities and pdfs for the null distributions (also for the theoretical distributions, but we'll overwrite these next)
                    dens_means = dens_k.mean(axis=-1).sum(axis=0) # (nvalid_species, nvalid_species, nmethods)
                    dens_stds = np.sqrt(dens_k.var(axis=-1).sum(axis=0)) # (nvalid_species, nvalid_species, nmethods)
                    pdf_means = pdf_bk.mean(axis=-1) # (nbins, k_max, nvalid_species, nvalid_species, nmethods)
                    pdf_stds = pdf_bk.std(axis=-1) # (nbins, k_max, nvalid_species, nvalid_species, nmethods)
                    #####################################################################################


                    #### Calculate the theoretical distribution-based means and stds ####################
                    # Ensure the dependencies of the *_check arrays are what we'd expect
                    # Should fail when nvalid_species==1, but just create an if-then statement... done
                    if nvalid_species != 1:
                        if not all_coords_are_same:
                            check_for_irrelevance(nvalid_holder_check, [False,False,True], 'nvalid_holder')
                        else:
                            check_for_irrelevance(nvalid_holder_check, [False,True,True], 'nvalid_holder')
                        if neighbor_freq_in_roi.var(axis=0) > tol:
                            check_for_irrelevance(nexpected_holder_check, [False,True,False], 'nexpected_holder')
                            check_for_irrelevance(pois_holder_check, [False,False,True,False], 'pois_holder')
                        else:
                            check_for_irrelevance(nexpected_holder_check, [False,True,True], 'nexpected_holder')
                            check_for_irrelevance(pois_holder_check, [False,False,True,True], 'pois_holder')
                    else:
                        check_for_irrelevance(nvalid_holder_check, [False,True,True], 'nvalid_holder')
                        check_for_irrelevance(nexpected_holder_check, [False,True,True], 'nexpected_holder')
                        check_for_irrelevance(pois_holder_check, [False,False,True,True], 'pois_holder')

                    # Assign the values needed for the fomulas to short-named variables, similar to the names in the formulas
                    L_k    = nvalid_holder                                                   # (       k_max, nvalid_species                )
                    rate_k = nexpected_holder                                                # (       k_max,                 nvalid_species)
                    f_bk   = pois_holder                                                     # (nbins, k_max,                 nvalid_species)

                    # Reshape the main arrays as needed to calculate the theoretical distributions
                    rate_k2 = rate_k.reshape((k_max,1,nvalid_species)) # (k_max, 1, nvalid_species)
                    L_k2 = L_k.reshape((k_max,nvalid_species,1)) # (k_max, nvalid_species, 1)
                    f_bk3 = f_bk.reshape((nbins,k_max,1,nvalid_species)) # (nbins,k_max,1,nvalid_species)
                    L_k3 = L_k.reshape((1,k_max,nvalid_species,1)) # (1,k_max,nvalid_species,1)

                    # Calculate the means and stds for the densities and pdfs for the theoretical distributions
                    dens_means[:,:,0] = rate_k2.sum(axis=0)
                    dens_stds[:,:,0] = np.sqrt((rate_k2/L_k2).sum(axis=0))
                    pdf_means[:,:,:,:,0] = f_bk3
                    pdf_stds[:,:,:,:,0] = np.sqrt((f_bk3-(f_bk3**2))/L_k3)
                    #####################################################################################


                    # Ensure new way compares with old way for the means of the theoretical distributions
                    mean_pdfs_check = np.zeros((nbins, k_max, nvalid_species, nvalid_species))
                    mean_densities_check = np.zeros((nvalid_species, nvalid_species))
                    for ineighbor_spec in range(nvalid_species):
                        neighbor_density = np.sum(species_roi==valid_species[ineighbor_spec], axis=0) / np.prod(roi_size)
                        mean_pdfs_check[:,:,:,ineighbor_spec] = np.expand_dims(calculate_pdf_theor(neighbor_density, areas, edges), axis=2) # note broadcasting is done here over the third index of mean_pdfs_roi
                        mean_densities_check[:,ineighbor_spec] = neighbor_density * (np.pi*(dr*k_max)**2)
                    if (np.abs(pdf_means[:,:,:,:,0]-mean_pdfs_check).max() > tol) or (np.abs(dens_means[:,:,0]-mean_densities_check).max() > tol):
                        print('ERROR: Old way of calculating theoretical means is different from new way')
                        print(np.abs(pdf_means[:,:,:,:,0]-mean_pdfs_check).max(), np.abs(dens_means[:,:,0]-mean_densities_check).max())
                        exit()
                    
            # Save the properties of the null distributions
                    null_properties_by_roi.append([(dens_means, dens_stds), (pdf_means, pdf_stds)])
                null_properties_by_slide.append(null_properties_by_roi)
            make_pickle(null_properties_by_slide, self.pickle_dir, pickle_file)

        else:

            null_properties_by_slide = self.load_pickle_class(pickle_file)

        self.null_properties = null_properties_by_slide


    # Calculate the pdfs for every ROI in every slide for every center-neighbor species pair and for every radius using the global nneighbors edges
    def calculate_pdfs(self, edges=None):

        # Import relevant module
        import numpy as np

        # Set variables already defined as attributes
        data_by_slide = self.data_by_slide
        if edges is None:
            edges = self.edges
        else:
            self.edges = edges
            self.midpoints = np.diff(edges)/2 + edges[:-1]
        k_max = self.k_max

        # Constant
        tol = 1e-7

        # Variable
        nbins = len(edges) - 1

        # For every slide...
        min_L = 1000
        all_pdfs = []
        for islide, data_by_roi in enumerate(data_by_slide):
            print('On slide {} of {}...'.format(islide+1, len(data_by_slide)))
            [_, _, roi_data] = data_by_roi

            # For every ROI in the slide...
            pdfs_slide = []
            for iroi, roi_data_item in enumerate(roi_data):
                print('  On ROI {} of {}...'.format(iroi+1, len(roi_data)))
                [_, _, _, _, _, _, _, _, _, _, _, valid_centers, nneighbors, _, _, nvalid_species, _] = roi_data_item

                # Initialize the array of interest
                pdfs_roi = np.zeros((nbins, k_max, nvalid_species, nvalid_species))

                # For every center species...
                for icenter_spec in range(nvalid_species):

                    # For every neighbor species...
                    for ineighbor_spec in range(nvalid_species):

                        # For every radial slice...
                        for ik in range(k_max):

                            # Calculate the actual number of neighbors around the centers
                            valid_centers3 = valid_centers[:,ik,icenter_spec] # centers of species icenter_spec that are valid at radius ik
                            to_histogram = nneighbors[valid_centers3,ik,ineighbor_spec] # before we replaced nneighbors2 in this line with pop_dens2

                            # Ensure the bins we're using are large enough
                            if to_histogram.max() >= edges[-1]:
                                print('ERROR: Bins are too low for histogrammed data')
                                print(to_histogram.max(), edges[-1])
                                exit()

                            # Calculate the normalized PDF
                            curr_pdf = np.histogram(to_histogram, bins=edges, density=True)[0]

                            # Ensure the histogram is normalized (technically we need to include the edges, but the bins are of size 1 so it doesn't matter)
                            if abs(np.sum(curr_pdf, axis=0)-1) > tol:
                                print('ERROR: Histogram doesn''t seem to be normalized')
                                exit()

                            # Ensure that numpy's way of calculating the PDF is how we would do it, essentially dividing the histogram by the number of elements in to_histogram
                            L = len(to_histogram)
                            h = curr_pdf*L
                            if np.max(np.abs(h-np.round(h))) > tol:
                                print('ERROR: Something is wrong with the histogram')
                                exit()

                            # Store the minimum number of elements to histogram... it should be at least min_nvalid_centers
                            if L < min_L:
                                min_L = L

                            # Populate the array of interest
                            pdfs_roi[:,ik,icenter_spec,ineighbor_spec] = curr_pdf

        # Set the calculated attribute, pdfs
                pdfs_slide.append(pdfs_roi) # save the ROI pdfs
            all_pdfs.append(pdfs_slide) # save the slide pdfs
        self.pdfs = all_pdfs

        print('Min L: {}'.format(min_L))


    # Calculate the number of neighbors and valid centers (as well as some other data) for every ROI in every slide
    # Save all this data in neighbors_counts.pkl; if this file already exists, just load the data
    def count_neighbors(self, k_max=10, dr=8, min_nvalid_centers=50):

        # Import relevant modules
        import os
        import numpy as np

        # Set variables already defined as attributes
        unique_slides = self.unique_slides
        pickle_dir = self.pickle_dir

        # Constant
        pickle_file = 'neighbors_counts.pkl'

        # If the pickle file doesn't exist...
        if not os.path.exists(os.path.join(pickle_dir,pickle_file)):

            # For each slide...
            data_by_slide = []
            for uslide in unique_slides:
                print('On slide '+uslide+'...')

                # Get the unique ROIs in the current slide
                unique_rois = np.unique(self.data.iloc[np.nonzero((self.data['Slide ID']==uslide).to_numpy())[0]]['tag'])

                # For each ROI in the slide...
                roi_data = []
                for uroi in unique_rois:
                    print('  On ROI '+uroi+'...')

                    # Get the needed ROI data
                    data_roi = self.data.iloc[np.nonzero((self.data['tag']==uroi).to_numpy())[0]]
                    data_roi = data_roi.reset_index(drop=True)
                    x_roi = np.array(data_roi['Cell X Position'])
                    y_roi = np.array(data_roi['Cell Y Position'])
                    species_roi = np.array(data_roi['Species int'], dtype='uint64')

                    # Do some checks and outputting
                    roi_checks_and_output(x_roi, y_roi)

                    # Get the numbers of neighbors and valid_centers, both for just the valid center species, for the current ROI (also get the population density)
                    x_roi, y_roi, x_range, y_range, roi_size, ncells, unique_species_roi, nunique_species_roi, ind_all, valid_centers, nneighbors, area, pop_dens, valid_species, nvalid_species, last_nvalid_centers, k_max, dr, min_nvalid_centers, valid_center_species_ind = self.count_neighbors_in_roi_class(x_roi, y_roi, species_roi, k_max=k_max, dr=dr, min_nvalid_centers=min_nvalid_centers)
                    
                    # Save the ROI data
                    roi_data_item = [x_roi, y_roi, species_roi, roi_size, x_range, y_range, ncells, unique_species_roi, nunique_species_roi, ind_all, pop_dens, valid_centers, nneighbors, area, valid_species, nvalid_species, valid_center_species_ind] # valid species is in decreasing frequency order and nvalid_species is its length
                    roi_data.append(roi_data_item)

            # Save the current slide data and the inputted parameters
                data_by_slide.append([uslide, unique_rois, roi_data])
            self.make_pickle_dict(['data_by_slide', 'k_max', 'dr', 'min_nvalid_centers'], locals(), pickle_file)

        # If the pickle file already exists, load it
        else:
            self.load_pickle_dict(pickle_file)

        # Calculate some basic slide properties (maximum area in all of the data, and the coordinate spacing)
        self.get_slide_data_properties()


    # This is just an implementation of count_neighbors_in_roi() that is part of the TIMECellInteraction class
    def count_neighbors_in_roi_class(self, x_roi, y_roi, species_roi, do_printing=True, k_max=None, dr=None, min_nvalid_centers=None):
        
        # Set variables already defined as attributes
        if k_max is None:
            k_max = self.k_max
        if dr is None:
            dr = self.dr
        if min_nvalid_centers is None:
            min_nvalid_centers = self.min_nvalid_centers

        return(count_neighbors_in_roi(x_roi, y_roi, species_roi, k_max, dr, min_nvalid_centers, do_printing, self.unique_species))


    # Get the maximum ROI area in all of the slide/ROI data, as well as the coordinate spacing
    def get_slide_data_properties(self):
        import numpy as np
        data_by_slide = self.data_by_slide
        max_area = 0
        coord_spacing = -1
        for data_by_roi in data_by_slide:
            [_, _, roi_data] = data_by_roi
            for roi_data_item in roi_data:
                [x_roi, _, _, roi_size, _, _, _, _, _, _, _, _, _, _, _, _, _] = roi_data_item
                curr_area = np.prod(roi_size)
                if curr_area > max_area:
                    max_area = curr_area
                x_roi2 = x_roi.copy()
                x_roi2.sort()
                curr_coord_spacing = np.unique(x_roi2[1:]-x_roi2[0:-1])[1]
                if coord_spacing == -1:
                    coord_spacing = curr_coord_spacing
                else:
                    if curr_coord_spacing != coord_spacing:
                        print('ERROR: Inconsistent coordinate spacing found')
                        exit()
        print('Maximum area calculated:', max_area)
        print('Coordinate spacing calculated:', coord_spacing)
        print('Note: The coord_spacing value above ({}) is the number of microns in each spatial unit in the (x,y)-coordinates.'.format(coord_spacing))
        print('      I.e., units are {} microns (u).'.format(coord_spacing))
        print('      E.g., if an arbitrary value of the x-coordinate is {}, then this corresponds to {} * {} = {} u.'.format(400, coord_spacing, 400, int(coord_spacing*400)))
        self.max_area = max_area
        self.coord_spacing = coord_spacing


    # Load some data from a pickle file ("class" just refers to this function being part of the TIMECellInteraction class)
    def load_pickle_class(self, pickle_file, pickle_dir=None):
        if pickle_dir is None:
            pickle_dir = self.pickle_dir
        return(load_pickle(pickle_dir, pickle_file))


    # Load a bunch of values to the self object from a pickle file by way of a dictionary
    def load_pickle_dict(self, pickle_file, pickle_dir=None):
        dict2load = self.load_pickle_class(pickle_file, pickle_dir=pickle_dir)
        for key in dict2load:
            val = dict2load[key]
            setattr(self, key, val)


    # Make a pickle file of a dictionary of data
    def make_pickle_dict(self, vars2save, local_dict, pickle_file):
        dict2save = {}
        for key in vars2save:
            if key in local_dict:
                val = local_dict[key]
                setattr(self, key, val)
            else:
                val = getattr(self, key)
            dict2save.update({key: val})
        make_pickle(dict2save, self.pickle_dir, pickle_file)


    # Plot all the results for the current dataset: the scatterplot ROI itself, density, and PDF; for the latter two, for each null distribution, plot the raw values, the null means, the demeaned values, and the z-scores
    # Save this all to a webpage that plots all the results for the dataset
    def plot_results(self, marker_size_step, resultsdir, dpi, myfontscale, dens_range, dens_demeaned_range, pdf_range, pdf_demeaned_range, z_score_range, alpha, mapping_dict):

        # Antonio's fix to enable plot generation in SLURM's batch mode
        import matplotlib
        matplotlib.use('Agg')        
        
        # Import relevant modules
        import matplotlib.pyplot as plt
        import matplotlib, os, functools

        # Set variables already defined as attributes
        slides = self.data_by_slide
        dens_by_slide = self.density_tot
        pdfs_by_slide = self.pdfs
        null_by_slide = self.null_properties
        plotting_map = self.plotting_map
        num_colors = self.num_colors

        # Constants
        color_cycle = matplotlib.rcParams['axes.prop_cycle']()
        default_marker_size = matplotlib.rcParams['lines.markersize']
        tol = 1e-8
        do_plot = True

        # Extract the correct number of colors from the default color palette
        ielem = 0
        colors = []
        for elem in color_cycle:
            color = elem['color']
            colors.append(color)
            ielem = ielem + 1
            if ielem == num_colors:
                break

        # Initialize the figure windows
        fig_roi = plt.subplots(figsize=(6,4))[0]
        fig_pdfs = plt.subplots(figsize=(6,6))[0]
        fig_dens = plt.subplots(figsize=(4,4))[0]

        # Create the results directory if it doesn't already exist
        if not os.path.exists(resultsdir):
            os.makedirs(resultsdir)

        # Initialize the HTML file
        f = open(os.path.join(resultsdir, 'analysis.html'), 'w')
        f.write('<html><head><title>data</title></head><body>\n')
        
        # Output the heatmap plotting ranges
        f.write('Fixed heatmap plotting ranges:\n')
        f.write('<ul>\n')
        f.write('<li>Density range: {}</li>\n'.format(dens_range))
        f.write('<li>Density de-meaned range: {}</li>\n'.format(dens_demeaned_range))
        f.write('<li>PDF range: {}</li>\n'.format(pdf_range))
        f.write('<li>PDF de-meaned range: {}</li>\n'.format(pdf_demeaned_range))
        f.write('<li>Z-score range: {}</li>\n'.format(z_score_range))
        f.write('</ul>\n')

        # Save the names of the null distribution methods
        null_distrib_method_names = ['theoretical Poisson', 'center=rand, neighbor=rand', 'center=rand, neighbor=foot', 'center=orig, neighbor=rand', 'center=orig, neighbor=foot']
            
        # For every set of slide data...
        for slide, dens_by_roi, pdfs_by_roi, null_by_roi in zip(slides, dens_by_slide, pdfs_by_slide, null_by_slide):
            [uslide, unique_rois, rois] = slide
            print('On slide '+uslide+'...')

            # For every set of ROI data within the slide data...
            for roi, dens, pdfs, [dens_null, pdfs_null], uroi in zip(rois, dens_by_roi, pdfs_by_roi, null_by_roi, unique_rois):
                [x_roi, y_roi, species_roi, _, x_range, y_range, _, unique_species_roi, _, _, _, _, _, _, valid_species, _, _] = roi # valid species is in decreasing frequency order and nvalid_species is its length
                print('  On ROI '+uroi+'...')

                # Define some variables for brief calculations and plotting
                spec2plot_roi = [x[0] for x in plotting_map if x[0] in unique_species_roi]
                spec_elements = [x[1] for x in plotting_map if x[0] in valid_species]
                pretty_names = [get_descriptive_cell_label(x, mapping_dict)[0] for x in spec_elements]
                filepath_roi = os.path.join(resultsdir, uroi+'_roi.png')
                filepath_pdfs = os.path.join(resultsdir, uroi+'_pdfs')
                filepath_dens = os.path.join(resultsdir, uroi+'_dens')
                mean_densities_roi, std_densities_roi = dens_null
                mean_pdfs_roi, std_pdfs_roi = pdfs_null
                nmethods = mean_densities_roi.shape[-1]

                # Plot the ROI and the density heatmap in one row
                f.write('<table border=0><tr>\n')
                f.write('<td align="center">\n')
                plot_roi(fig_roi, spec2plot_roi, species_roi, x_roi, y_roi, plotting_map, colors, x_range, y_range, uroi, marker_size_step, default_marker_size, filepath_roi, dpi, mapping_dict, self.coord_spacing, do_plot=do_plot, alpha=alpha)
                f.write('<img src="'+os.path.basename(filepath_roi)+'">\n')
                f.write('</td>\n')
                plot_and_write_dens(image=dens, imrange=dens_range, title='Density', filepath=filepath_dens+'.png', fig=fig_dens, f=f, pretty_names=pretty_names, dpi=dpi, uroi=uroi, do_plot=do_plot)
                f.write('</tr></table>\n')

                # Plot the null means, de-meaned density, and z-score in one row for each null distribution method
                for imethod in range(nmethods):
                    means = mean_densities_roi[:,:,imethod]
                    stds = std_densities_roi[:,:,imethod]
                    if do_plot:
                        [dens], [means, stds] = expand_nbins([dens], [means, stds]) # ensure subtraction of same-sized arrays, padding one of the two lists with zeros if necessary
                    else:
                        import numpy as np
                        dens = np.zeros(means.shape, dtype='uint8')
                    f.write('<h2>Null distribution type: {}</h2>\n'.format(null_distrib_method_names[imethod]))
                    f.write('<table border=0><tr>\n')

                    # Create figures of the null means, demeaned density, and density z-score for the inputted ROI and write <TD> HTML elements for each of these images
                    plot_and_write_dens_partial = functools.partial(plot_and_write_dens, fig=fig_dens, f=f, pretty_names=pretty_names, dpi=dpi, uroi=uroi, imethod=imethod, do_plot=do_plot) # fill in some of the arguments to plot_and_write_dens() in order to create a smaller function plot_and_write_dens_partial() that can be called more succinctly
                    plot_and_write_dens_partial(image=means, imrange=dens_range, title='Density null means', filepath=filepath_dens+'_method-{}_means.png'.format(imethod))
                    plot_and_write_dens_partial(image=dens-means, imrange=dens_demeaned_range, title='Density less null means', filepath=filepath_dens+'_method-{}_demeaned.png'.format(imethod))
                    plot_and_write_dens_partial(image=(dens-means)/(stds+tol), imrange=z_score_range, title='Density z-scores', filepath=filepath_dens+'_method-{}_z-scores.png'.format(imethod))
                    f.write('</tr></table>\n')

                # Plot the ROI and the PDF heatmap grid in one row
                f.write('<table border=0><tr>\n')
                f.write('<td align="center">\n')
                f.write('<img src="'+os.path.basename(filepath_roi)+'">\n')
                f.write('</td>\n')
                plot_and_write_pdfs(image=pdfs, imrange=pdf_range, title='PDFs', filepath=filepath_pdfs+'.png', fig=fig_pdfs, f=f, spec_elements=spec_elements, dpi=dpi, uroi=uroi, myfontscale=myfontscale, do_plot=do_plot)
                f.write('</tr></table>\n')

                # Plot the null means, de-meaned PDFs, and z-score in one row for each null distribution method
                for imethod in range(nmethods):
                    means = mean_pdfs_roi[:,:,:,:,imethod]
                    stds = std_pdfs_roi[:,:,:,:,imethod]
                    if do_plot:
                        [pdfs], [means, stds] = expand_nbins([pdfs], [means, stds]) # ensure subtraction of same-sized arrays, padding one of the two lists with zeros if necessary
                    else:
                        import numpy as np
                        pdfs = np.zeros(means.shape, dtype='uint8')
                    f.write('<h2>Null distribution type: {}</h2>\n'.format(null_distrib_method_names[imethod]))
                    f.write('<table border=0><tr>\n')

                    # Create figures of grids of the null means, demeaned PDFs, and PDF z-scores for the inputted ROI and write <TD> HTML elements for each of these grid images
                    plot_and_write_pdfs_partial = functools.partial(plot_and_write_pdfs, fig=fig_pdfs, f=f, spec_elements=spec_elements, dpi=dpi, uroi=uroi, imethod=imethod, myfontscale=myfontscale, do_plot=do_plot)
                    plot_and_write_pdfs_partial(image=means, imrange=pdf_range, title='PDFs null means', filepath=filepath_pdfs+'_method-{}_means.png'.format(imethod))
                    plot_and_write_pdfs_partial(image=pdfs-means, imrange=pdf_demeaned_range, title='PDFs less null means', filepath=filepath_pdfs+'_method-{}_demeaned.png'.format(imethod))
                    plot_and_write_pdfs_partial(image=(pdfs-means)/(stds+tol), imrange=z_score_range, title='PDFs z-scores', filepath=filepath_pdfs+'_method-{}_z-scores.png'.format(imethod))
                    f.write('</tr></table>\n')

        # Finalize the HTML file
        f.write('</body></html>\n')
        f.close()


    # Preprocess the initial Pandas dataframe from Consolidata_data.txt (or a simulated one for simulated data) by creating another column (Species int) specifying a unique integer identifying the cell type
    # If requested, remove compound species, and return the list of single-protein "phenotypes" contained in the data
    def preprocess_dataframe(self, allow_compound_species):

        # Import relevant module
        import numpy as np

        # Preprocess the pandas dataframe in various ways
        data_phenotypes = self.data.filter(regex='^[pP]henotype ') # get just the "Phenotype " columns
        data_phenotypes = data_phenotypes.reset_index(drop=True)
        phenotype_cols = list(data_phenotypes.columns) # get a list of those column names
        phenotypes = np.array([x.replace('Phenotype ','') for x in phenotype_cols]) # extract just the phenotypes from that list
        n_phenotypes = len(phenotypes) # get the number of possible phenotypes in the datafile
        self.data['Species string'] = data_phenotypes.applymap(lambda x: '1' if str(x)[-1]=='+' else '0').apply(lambda x: ''.join(list(x)), axis=1) # add a column to the original data that tells us the unique "binary" species string of the species corresponding corresponding to that row/cell
        self.data = self.data.drop(np.nonzero((self.data['Species string'] == '0'*n_phenotypes).to_numpy())[0]) # delete rows that all have '...-' as the phenotype or are blank
        self.data = self.data.reset_index(drop=True) # reset the indices
        self.data['Species int'] = self.data['Species string'].apply(lambda x: int(x, base=2)) # add an INTEGER species column

        # Remove compound species if requested
        if not allow_compound_species:
            self.remove_compound_species()
            self.remove_compound_species() # ensure nothing happens

        return(phenotypes)


    # For each compound species ('Species int' not just a plain power of two), add each individual phenotype to the end of the dataframe individually and then delete the original compound entry
    def remove_compound_species(self):

        # Import relevant module
        import numpy as np

        # Get the species IDs
        x = np.array(self.data['Species int'])
        print('Data size:', len(self.data))

        # Determine which are not powers of 2, i.e., are compound species
        powers = np.log2(x)
        compound_loc = np.nonzero(powers != np.round(powers))[0]
        ncompound = len(compound_loc)
        print('  Compound species found:', ncompound)

        # If compound species exist...
        if ncompound > 0:

            print('  Removing compound species from the dataframe...')

            # Get a list of tuples each corresponding to compound species, the first element of which is the row of the compound species, and the second of which is the species IDs of the pure phenotypes that make up the compound species
            compound_entries = [(cl, 2**np.nonzero([int(y) for y in bin(x[cl])[2:]][-1::-1])[0]) for cl in compound_loc]
            
            # For each compound species...
            data_to_add = []
            for index, subspecies in compound_entries:

                # For each pure phenotype making up the compound species...
                for subspec in subspecies:

                    # You have to put this here instead of outside this loop for some weird reason! Even though you can see the correct change made to series and EVEN TO DATA_TO_ADD by looking at series and data_to_add[-1] below, for some Godforsaken reason the actual data_to_add list does not get updated with the change to 'Species int' when you print data_to_add at the end of both these loops, and therefore the data that gets added to the data dataframe contains all the same 'Species string' values, namely the last one assigned. Thus, we are actually adding the SAME species to multiple (usually 2) spatial points, so that the even-spacing problem arises.
                    series = self.data.iloc[index].copy()

                    series['Species int'] = subspec # set the species ID of the series data to that of the current phenotype
                    data_to_add.append(series) # add the data to a running list
                    
            # Add all the data in the list to the dataframe                
            self.data = self.data.append(data_to_add, ignore_index=True)
            print('  Added rows:', len(data_to_add))
            print('  Data size:', len(self.data))

            # Delete the original compound species entries
            self.data = self.data.drop(compound_loc)
            self.data = self.data.reset_index(drop=True)
            print('  Deleted rows:', len(compound_loc))
            print('  Data size:', len(self.data))


# Choose the most comprehensive nneighbors edges and, optionally, the smallest population densities, for all inputted datasets
def calculate_comprehensive_edges(edges, midpoints_pop_dens=None):
    import numpy as np
    comprehensive_edges = edges[np.argmax([len(x) for x in edges])]
    if midpoints_pop_dens is not None:
        smallest_pop_dens = midpoints_pop_dens[np.argmin([x[-1] for x in midpoints_pop_dens])]
        return(comprehensive_edges, smallest_pop_dens)
    else:
        return(comprehensive_edges)


# Return a Poisson distribution for the given edges for each area (slice) given an overall expected density
def calculate_pdf_theor(density, areas, edges):

    # Import relevant module
    import numpy as np

    # Dependent variables
    nbins = len(edges)-1
    nareas = len(areas)
    midpoints = np.diff(edges)/2 + edges[:-1]

    # Define the theoretical PDFs
    pdfs_theor = np.zeros((nbins, nareas))

    # Calculate the PDF for each area
    for iarea in range(nareas):
        nexpected = density * areas[iarea]
        #pdfs_theor[:,iarea] = normalize_hist(poisson(nexpected, midpoints), edges)
        pdfs_theor[:,iarea] = poisson(nexpected, midpoints)

    return(pdfs_theor)


# Check whether each axis in arr is a constant value and therefore contributes nothing (it's "irrelevant")
# Compare to a list of whether you'd expect each axis to be constant (should be constant)
def check_for_irrelevance(arr, should_be_constant, arr_name):
    import numpy as np
    tol = 1e-8
    constant_axes = []
    for axis in range(arr.ndim):
        mins = arr.min(axis=axis)
        maxs = arr.max(axis=axis)
        constant_axes.append(np.abs(maxs-mins).max()<tol)
    if constant_axes != should_be_constant:
        print('ERROR: Irrelevance check failed for '+arr_name)
        print(constant_axes, should_be_constant)
        print(arr)
        exit()


# For the inputted ROI, calculate the number of neighbors around each center for each neighbor species and for each radius
# In addition, calculate whether each center is valid for each radius and center species
# For both these quantities only return the data for the species that are valid centers in the ROI
def count_neighbors_in_roi(x_roi, y_roi, species_roi, k_max, dr, min_nvalid_centers, do_printing, unique_species):

    # Import relevant modules
    import numpy as np
    import scipy.spatial as spat

    # Calculate some dependent variables
    r_k = (np.arange(k_max)+1) * dr # calculate the outer radii of the thick rings
    nunique_species = len(unique_species)

    # Center the coordinates for the ROI and store the ROI edges
    roi_size = np.array([max(x_roi)-min(x_roi), max(y_roi)-min(y_roi)])
    x_range = np.array([-1,1]) * roi_size[0]/2
    y_range = np.array([-1,1]) * roi_size[1]/2
    x_roi = x_roi - min(x_roi) - x_range[1]
    y_roi = y_roi - min(y_roi) - y_range[1]

    # Get additional ROI data
    ncells = len(x_roi) # number of cells in the current ROI, i.e., C
    unique_species_roi = np.unique(species_roi) # unique species in the current ROI
    nunique_species_roi = len(unique_species_roi) # number of unique species in the current ROI, i.e., S

    # Calculate temporary quantities needed for populating the arrays of interest: pop_dens and valid_centers
    coords = np.array([x_roi,y_roi]).T
    dist_mat = spat.distance_matrix(coords,coords)
    r = np.tile(r_k.reshape((1,1,k_max)),(ncells,ncells,1)) - np.tile(dist_mat.reshape((ncells,ncells,1)),(1,1,k_max))
    H = np.pad(np.array(r>0, dtype='uint8'), ((0,0),(0,0),(1,0)), 'constant')

    # Get the plotting indices (in the same order) of the species that are actually present in the current ROI
    ind_all = [(np.nonzero(unique_species==y))[0][0] for y in filter(lambda x: x in unique_species_roi, unique_species)]

    # Initialize the arrays of interest
    nneighbors = np.zeros((ncells, k_max, nunique_species))
    valid_centers = np.zeros((ncells, k_max, nunique_species), dtype='bool')

    # Populate the area array
    area = np.tile((np.pi*dr*(2*r_k-dr)).reshape((1,k_max,1)), (ncells,1,nunique_species))

    # For each radial slice...
    for k0 in range(k_max):
        if do_printing:
            print('    On radius '+str(k0+1)+' of '+str(k_max)+'...')

        k = k0 + 1 # get the current k value
        c = H[:,:,k] - H[:,:,k-1] # calculate the current c matrix which is whether the neighbors are located within the current rings for the centers
        valid_center_location = (x_roi >= (x_range[0]+k*dr)) & (x_roi <= (x_range[1]-k*dr)) & (y_roi >= (y_range[0]+k*dr)) & (y_roi <= (y_range[1]-k*dr)) # determine whether the possible centers are valid to even count

        # For each unique species in the slide in decreasing frequency order (this represents both the center [s1] and neighbor [s2] species)...
        for is0, s0 in enumerate(ind_all):
            if do_printing:
                print('      On species '+str(is0+1)+' of '+str(nunique_species_roi)+'...')

            spec = unique_species[s0] # get the current species (both center and neighbor)
            species_eq_spec = species_roi == spec # determine whether the possible species equal the current center or neighbor
            valid_centers[:,k0,s0] = valid_center_location & species_eq_spec # determine which centers are valid

            # For each possible center and neighbor, if the current neighbor is of the current species, then count cell j as a neighbor of cell i, and
            # if the current center is of the desired type for counting the neighbors, then subtract its contribution since we shouldn't count the center as being a neighbor of itself
            nneighbors[:,k0,s0] = np.sum(np.tile(species_eq_spec.reshape((ncells,1)),(1,ncells))*c,axis=0) - species_eq_spec*np.diag(c)

    # Calculate the population density
    pop_dens = nneighbors / area

    # Get the indices in decreasing frequency order of the centers who number > min_nvalid_centers for the largest radius
    last_nvalid_centers = np.sum(valid_centers, axis=0)[-1,:] # counts of the number of valid centers for each species in the largest slice (nunique_species,)
    valid_center_species_ind = np.nonzero((last_nvalid_centers>min_nvalid_centers))[0] # indexes of which unique species have a minimal number of valid centers (goes into something of size nunique_species)
    last_nvalid_centers = last_nvalid_centers[valid_center_species_ind] # counts of the number of valid centers for just the valid (minimally numbering) unique species
    valid_center_species = unique_species[valid_center_species_ind] # the IDs of the valid unique species
    nneighbors = nneighbors[:,:,valid_center_species_ind] # nneighbors for the valid unique species
    valid_centers = valid_centers[:,:,valid_center_species_ind] # valid_centers for the valid unique species
    area = area[:,:,valid_center_species_ind] # area for the valid unique species
    pop_dens = pop_dens[:,:,valid_center_species_ind] # pop_dens for the valid unique species
    nvalid_center_species = len(valid_center_species) # number of valid unique species

    return(x_roi, y_roi, x_range, y_range, roi_size, ncells, unique_species_roi, nunique_species_roi, ind_all, valid_centers, nneighbors, area, pop_dens, valid_center_species, nvalid_center_species, last_nvalid_centers, k_max, dr, min_nvalid_centers, valid_center_species_ind) # the latter (last_nvalid_centers) is the same as the typical nvalid_species
    # note that valid_center_species is in decreasing frequency order, which makes sense because it's made of up something (unique_species) that's in decreasing frequency order that is indexed by something (valid_center_species_ind) that is monotonic


# Ensure two lists are the same size, appending zeros to the smaller of the two lists
def expand_nbins(list1, list2):

    # Import relevant module
    import numpy as np

    # Variables
    nbins1 = list1[0].shape[0]
    nbins2 = list2[0].shape[0]

    # If one of the lists has arrays that have a different number of bins (first index)...
    if nbins1 != nbins2:

        # Ensure the list with the larger nbins is list2
        do_swap = False
        if nbins1 > nbins2:
            do_swap = True
            list1, list2 = list2, list1
            nbins1, nbins2 = nbins2, nbins1

        # Expand nbins of the items in list1
        shp = list(list1[0].shape)
        shp[0] = nbins2 - nbins1
        zeros_arr = np.zeros(tuple(shp))
        list1_new = []
        for item in list1:
            list1_new.append(np.r_[item, zeros_arr])
        list1 = list1_new

        # Unswap if necessary
        if do_swap:
            list1, list2 = list2, list1

    return(list1, list2)


# Get the average spacing on either side of each datapoint in an array
def get_avg_spacing(arr):
    if len(arr) >= 2:
        import numpy as np
        arr2 = np.concatenate(([2*arr[0]-arr[1]], arr, [2*arr[-1]-arr[-2]]))
        return((arr2[2:]-arr2[0:-2]) / 2)
    else:
        print('Not actually getting average spacing in arr because len(arr) < 2; returning 1')
        return([1])


# Read in the Consolidated_data.txt TSV file into a Pandas dataframe
def get_consolidated_data(csv_file):
    import pandas as pd
    return(pd.read_csv(csv_file, sep='\t')) # read in the data


# Given a list of phenotypes in a species, return the A+/B+ etc. string version
def phenotypes_to_string(phenotype_list):
    phenotype_list.sort()
    return('/'.join(phenotype_list))


# Given a list of phenotypes in a species, return the nicely formatted version, if there's a known cell type corresponding to the species
# Note these are purely labels; the species themselves are determined by allow_compound_species as usual
def get_descriptive_cell_label(phenotype_list, mapping_dict):
    # Note: CD163+/CD4+ REALLY ISN'T ANYTHING COMPOUND --> MAKE IT OVERLAPPING SPECIES (i.e., it shouldn't be in the dictionary below)!!!!
    phenotype_string = phenotypes_to_string(phenotype_list)
    descriptive_name = mapping_dict.get(phenotype_string)
    if descriptive_name is None:
        pretty_name = phenotype_string
        is_known = False
    else:
        pretty_name = phenotype_string + ' (' + descriptive_name + ')'
        is_known = True
    return(pretty_name, is_known)


# Obtain the plotting map, total number of unique colors needed for plotting, the list of unique species (in the same order as in plotting_map), and a correctly sorted list of slides (e.g., 1,2,15 instead of 1,15,2)
# Note that individual unique species are specified by the allow_compound_species keyword, which in turn affects which of the 'Species int' columns of the Pandas dataframe are actually unique
# Don't get confused by the mapping_dict variable, which only affects plotting of the species... it doesn't affect what is actually considered a unique species or not!
def get_dataframe_info(data, phenotypes, mapping_dict):

    # Import relevant modules
    import numpy as np
    from operator import itemgetter

    # Create an ndarray containing all the unique species in the dataset in descending order of frequency with columns: integer label, string list, frequency, color(s), circle size(s)
    plotting_map = [[-(list(data['Species int']).count(x)), list(int2list(phenotypes, x)), x] for x in np.unique(data['Species int'])] # create a list of the unique species in the dataset with columns: -frequency, string list, integer label
    plotting_map.sort(key=itemgetter(0,1)) # sort by decreasing frequency (since the frequency column is negative) and then by the string list
    plotting_map = [ [-x[0], x[1], x[2]] for x in plotting_map ] # make the frequency column positive

    print(plotting_map)

    # Get the colors of the species that are already known to us; use a -1 if the species isn't known
    colors = []
    known_phenotypes = []
    known_colors = []
    icolor = 0
    for item in plotting_map:
        phenotype_list = item[1]
        is_known = get_descriptive_cell_label(phenotype_list, mapping_dict)[1]

        # If the species (each row of plotting_map) is known to us (i.e., in the inputted mapping_dict variable, which simply assigns a cell label to any single or compound species)...
        # ...give that species its own color, and make a note if the species is also a single, non-compound species (i.e., a single phenotype)
        if is_known:
            colors.append(icolor)
            if len(phenotype_list) == 1:
                known_phenotypes.append(phenotype_list[0])
                known_colors.append(icolor)
            icolor = icolor + 1
        else:
            colors.append(-1)

    # Get the colors of the rest of the species using the colors of the already-known single-phenotype species
    # I.e., if the species is not known to us (i.e., not in mapping_dict), do not give the species its own color (unless it contains a phenotype that's not in known_phenotypes)
    # Instead, give each phenotype in the species either the corresponding color in known_phenotypes (if it's in there) or a new color (if it's not in known_phenotypes)
    # Assign the corresponding circle sizes as well
    colors2 = []
    circle_sizes = []
    for item, color in zip(plotting_map, colors):
        phenotype_list = item[1]
        if color == -1:
            curr_colors = []
            for single_phenotype in phenotype_list:
                if single_phenotype in known_phenotypes:
                    curr_colors.append(known_colors[known_phenotypes.index(single_phenotype)])
                else:
                    curr_colors.append(icolor)
                    known_phenotypes.append(single_phenotype)
                    known_colors.append(icolor)
                    icolor = icolor + 1
        else:
            curr_colors = [color]
        
        # Always have the most prevalent single species (if a lower color number really implies higher prevalence, it should generally at least) first in the color list, and make the corresponding circle size the largest (but in the background of course)
        curr_colors.sort()
        curr_sizes = list(np.arange(start=len(curr_colors), stop=0, step=-1))

        colors2.append(curr_colors)
        circle_sizes.append(curr_sizes)
    colors = colors2

    # Store the total number of unique colors to plot
    num_colors = icolor

    # Finalize the plotting map
    plotting_map = np.array([ [item[2], item[1], item[0], (color if len(color)!=1 else color[0]), (circle_size if len(circle_size)!=1 else circle_size[0])] for item, color, circle_size in zip(plotting_map, colors, circle_sizes) ])

    # Use the plotting map to extract just the unique species in the data
    unique_species = np.array([x[0] for x in plotting_map]) # get a list of all the unique species in the dataset in the correct order

    # Get the unique slides sorted correctly
    tmp = [[int(x.split('-')[0][0:len(x.split('-')[0])-1]),x] for x in np.unique(data['Slide ID'])]
    tmp.sort(key=(lambda x: x[0]))
    unique_slides = [x[1] for x in tmp]

    return(plotting_map, num_colors, unique_species, unique_slides)


# Given an array of densities, for each of which we will generate a different ROI of the corresponding density, return a Pandas dataframe of the simulated data
# Only a single slide will be returned (but in general with multiple ROIs)
def get_simulated_data(doubling_type, densities, max_real_area, coord_spacing, mult):

    # Import relevant modules
    import numpy as np
    import pandas as pd

    # Warn if we're doubling up the data
    if doubling_type != 0:
        print('NOTE: DOUBLING UP THE SIMULATED DATA!!!!')

    # Specify the columns that are needed based on what's used from consolidated_data.txt
    columns = ['tag', 'Cell X Position', 'Cell Y Position', 'Slide ID', 'Phenotype A', 'Phenotype B']

    # Determine a constant number of cells in a simulated ROI using the maximum-sized ROI of the real data and the largest simulated density
    N = int(np.int(densities[-1]*max_real_area) * mult)

    # Initialize the Pandas dataframe
    data = pd.DataFrame(columns=columns)

    # For each ROI density and average either-side density spacing...
    perc_error = []
    for pop_dens, avg_spacing in zip(densities, get_avg_spacing(densities)):
        print('Current population density:', pop_dens)

        # Get the Cartesian coordinates of all the cells from random values populated until the desired density is reached
        tot_area = N / pop_dens
        side_length = np.sqrt(tot_area)
        tmp = np.round(side_length/coord_spacing) # now we want N random integers in [0,tmp]
        coords_A = np.random.randint(tmp, size=(int(2*N/3), 2)) * coord_spacing
        coords_B = np.random.randint(tmp, size=(int(1*N/3), 2)) * coord_spacing
        x_A = coords_A[:,0]
        y_A = coords_A[:,1]
        x_B = coords_B[:,0]
        y_B = coords_B[:,1]

        # Set the ROI name from the current density
        tag = 'pop_dens_{:09.7f}'.format(pop_dens)

        # Add the simulated data to a nested list and then convert to a Pandas dataframe and add it to the master dataframe
        # Number in bracket is number of actually unique dataset (three actually unique datasets)
        # (1) [1] doubling_type=0, allow_compound_species=True:  coords=(A,B), labels=(A,B) # two species with different coordinates
        # (2) [1] doubling_type=0, allow_compound_species=False: coords=(A,B), labels=(A,B)
        # (3) [2] doubling_type=1, allow_compound_species=True:  coords=(A,A), labels=(A,B) # two overlapping species
        # (4) [2] doubling_type=1, allow_compound_species=False: coords=(A,A), labels=(A,B)
        # (5) [3] doubling_type=2, allow_compound_species=True:  coords=(A),   labels=(AB)  # one compound species (AB = compound species)
        # (6) [2] doubling_type=2, allow_compound_species=False: coords=(A,A), labels=(A,B)
        if doubling_type != 2:
            list_set = [ [tag, curr_x_A, curr_y_A, '1A-only_slide', 'A+', 'B-'] for curr_x_A, curr_y_A in zip(x_A, y_A) ]
            if doubling_type == 0:
                list_set = list_set + [ [tag, curr_x_B, curr_y_B, '1A-only_slide', 'A-', 'B+'] for curr_x_B, curr_y_B in zip(x_B, y_B) ]
            elif doubling_type == 1:
                list_set = list_set + [ [tag, curr_x_A, curr_y_A, '1A-only_slide', 'A-', 'B+'] for curr_x_A, curr_y_A in zip(x_A, y_A) ]
        else:
            list_set = [ [tag, curr_x_A, curr_y_A, '1A-only_slide', 'A+', 'B+'] for curr_x_A, curr_y_A in zip(x_A, y_A) ]
        tmp = pd.DataFrame(list_set, columns=columns)
        data = data.append(tmp, ignore_index=True)

        # Calculate the percent error in the actual density from the desired density
        if doubling_type == 0:
            x_tmp = np.r_[x_A,x_B]
            y_tmp = np.r_[y_A,y_B]
        else:
            x_tmp = x_A
            y_tmp = y_A
        perc_error.append((N / ((x_tmp.max()-x_tmp.min()) * (y_tmp.max()-y_tmp.min())) - pop_dens) / avg_spacing * 100)

    print('Percent error:', perc_error)
    print('Maximum percent error:', np.max(perc_error))

    return(data)


# Convert integer numbers defined in species bit-wise to a string list based on phenotypes
def int2list(phenotypes, species):
    return(phenotypes[[bool(int(char)) for char in ('{:0'+str(len(phenotypes))+'b}').format(species)]])


# Load some data from a pickle file
def load_pickle(pickle_dir, pickle_file):
    import pickle, os
    filename = os.path.join(pickle_dir, pickle_file)
    print('Reading pickle file '+filename+'...')
    with open(filename, 'rb') as f:
        data_to_load = pickle.load(f)
    return(data_to_load)


# Write a pickle file from some data
def make_pickle(data_to_save, pickle_dir, pickle_file):
    import pickle, os
    filename = os.path.join(pickle_dir, pickle_file)
    print('Creating pickle file '+filename+'...')
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)


# Normalize a histogram using its corresponding edges, turning it into a pdf
# Only apply this to an empirical histogram, never a theoretical one
def normalize_hist(hist, edges):
    import numpy as np
    return(hist / np.sum(hist*np.diff(edges)))


# Create a figure of a density heatmap for the inputted ROI and write a <TD> HTML element containing the image
def plot_and_write_dens(fig, f, image, imrange, pretty_names, filepath, dpi, title, uroi, imethod=None, do_plot=True):
    if do_plot:
        plot_dens(fig, image, imrange, pretty_names, filepath, dpi, title)
    write_html(f, uroi, title, filepath, imethod=imethod)


# Plot a grid of PDFs for the common species in the current ROI to a figure, and save the resulting grid into a single image, and write a <TD> HTML element containing the image
def plot_and_write_pdfs(fig, f, image, imrange, spec_elements, filepath, dpi, title, uroi, myfontscale, imethod=None, do_plot=True):
    if do_plot:
        plot_pdfs(fig, image, imrange, spec_elements, myfontscale, filepath, dpi)
    write_html(f, uroi, title, filepath, imethod=imethod)


# Plot a heatmap of the densities per species pair (for a single ROI) and save the figure
def plot_dens(fig, dens, dens_range, pretty_names, filepath, dpi, title):
    
    # Plot the heatmap
    plot_dens_heatmap(fig.axes[0], dens, dens_range, pretty_names, title)

    # Save the figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


# Plot a heatmap (from image) on a given axis (ax) and the image range (dens_range)
def plot_dens_heatmap(ax, image, dens_range, axis_labels, title):
    ax.cla()
    import numpy as np
    import matplotlib.pyplot as plt
    ax.imshow(image, vmin=dens_range[0], vmax=dens_range[1])
    ticks_arr = np.arange(len(axis_labels))
    ax.set_xticks(ticks_arr)
    ax.set_xticklabels(axis_labels)
    ax.set_yticks(ticks_arr)
    ax.set_yticklabels(axis_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Neighbor species') # k_max (old, but keeping here to show x and y are swapped)
    ax.set_ylabel('Center species') # nbins (old, but keeping here to show x and y are swapped)
    ax.set_title(title)


# Plot a grid of PDFs for the common species to an inputted figure ID and save the resulting grid into a single image
def plot_pdfs(fig, pdfs, pdfs_range, unique_species_list2, myfontscale, filepath, dpi):

    # Clear the figure
    fig.clf()

    # Variable
    nvalid_species = pdfs.shape[2]

    # Axes
    ax_pdfs = fig.subplots(nrows=nvalid_species, ncols=nvalid_species)

    # For every center-neighbor pair, plot the PDF
    for icenter_spec in range(nvalid_species):
        for ineighbor_spec in range(nvalid_species):
            if nvalid_species == 1:
                ax_pdf = ax_pdfs
            else:
                ax_pdf = ax_pdfs[icenter_spec][ineighbor_spec]
            plot_pdf_heatmap(ax_pdf, pdfs[:,:,icenter_spec,ineighbor_spec], pdfs_range, '+'.join(unique_species_list2[icenter_spec]) + ', ' + '+'.join(unique_species_list2[ineighbor_spec]))
            if icenter_spec == nvalid_species-1:
                ax_pdf.set_xlabel('Radius index') # k_max
            scale_text_in_plot(ax_pdf, myfontscale) # scale down the text on the axes

    # Save the figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


# Plot a single PDF heatmap (image) on an inputted axis
def plot_pdf_heatmap(ax, image, prob_dens_range, title):
    ax.cla()
    ax.imshow(image, vmin=prob_dens_range[0], vmax=prob_dens_range[1])
    ax.set_ylabel('Number of neighbors') # nbins
    ax.set_title(title)


# For the raw data (coordinates) for a given ROI, plot a circle (scatter plot) representing each species, whether known (in which case get_descriptive_cell_label() is used) or unknown; plot a legend too
# Save the figure to a file as well
def plot_roi(fig, spec2plot, species, x, y, plotting_map, colors, x_range, y_range, title, marker_size_step, default_marker_size, filepath, dpi, mapping_dict, coord_spacing=0.5, do_plot=True, alpha=1):

    if do_plot:

        # Import relevant library
        import numpy as np

        # Axis
        ax = fig.axes[0]
        ax.cla()

        # For each unique species in the current ROI (in the correct order)...
        plotted_colors = []
        plots_for_legend = []
        labels_for_legend = []
        for spec in spec2plot:

            # Get the data for that species
            spec_ind = np.nonzero(species==spec)
            x_pos = x[spec_ind]
            y_pos = y[spec_ind]
            plotting_data = plotting_map[[x[0] for x in plotting_map].index(spec)]

            # Ensure the colors and marker sizes are lists and determine whether a single circle will be plotted for the potentially compound species
            if not isinstance(plotting_data[3],list):
                all_colors = [plotting_data[3]]
                all_sizes = [plotting_data[4]]
                is_primary = True
            else:
                all_colors = plotting_data[3]
                all_sizes = plotting_data[4]
                is_primary = False

            # For each circle to plot within the current species...
            for icircle, (curr_color, curr_size) in enumerate(zip(all_colors, all_sizes)):

                # Obtain the actual color and marker size to plot
                curr_color2 = colors[curr_color]
                curr_size2 = (( (curr_size-1) *marker_size_step+1) * default_marker_size) ** 2

                # Plot the current species
                # Note: coord_spacing is # of microns per unit
                curr_plt = ax.scatter(x_pos*coord_spacing, y_pos*coord_spacing, s=curr_size2, c=curr_color2, edgecolors='k', alpha=alpha)

                # If we're on a primary species (in which a single circle is plotted for a potentially compound species), add the current plot to the legend
                if is_primary:
                    curr_label = get_descriptive_cell_label(plotting_data[1], mapping_dict)[0]
                    plotted_colors.append(curr_color) # keep a record of the colors we've plotted so far in order to add a minimal number of non-primary species to the legend
                    plots_for_legend.append(curr_plt)
                    labels_for_legend.append(curr_label)

                # If we're on a non-primary species, only add it to the legend if the color hasn't yet been plotted
                else:

                    # If the color hasn't yet been plotted...
                    if curr_color not in plotted_colors:

                        # Get the correct label for the current phenotype within the non-primary species
                        curr_label = get_descriptive_cell_label([plotting_data[1][icircle]], mapping_dict)[0]

                        # If the current color to add to the legend was NOT a minimal size, first make a dummy plot of one of the minimal size
                        if not curr_size == 1:
                            curr_plt = ax.scatter(x_range[0]*2*coord_spacing, y_range[0]*2*coord_spacing, s=(default_marker_size**2), c=curr_color2, edgecolors='k', alpha=alpha)

                        # Add the plot to the legend
                        plotted_colors.append(curr_color) # keep a record of the colors we've plotted so far in order to add a minimal number of non-primary species to the legend
                        plots_for_legend.append(curr_plt)
                        labels_for_legend.append(curr_label)

        # Complete the plot and save to disk
        ax.set_aspect('equal')
        ax.set_xlim(tuple(x_range*coord_spacing))
        ax.set_ylim(tuple(y_range*coord_spacing))
        ax.set_xlabel('X coordinate (microns)')
        ax.set_ylabel('Y coordinate (microns)')
        ax.set_title('ROI ' + title)
        ax.legend(plots_for_legend, labels_for_legend, loc='upper left', bbox_to_anchor=(1, 0, 1, 1))

        # Save the figure
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')


# Return the probability of a set of events given the average event rate
def poisson(rate, events):
    import numpy as np
    import scipy.special
    gamma = scipy.special.gamma
    return(rate**events * np.exp(-rate) / gamma(events+1))


# For the inputted ROI coordinates, ensure the coordinate spacing is what we expect it to be (hardcoded to 0.5) and print the coordinate range, number of cells, ROI area, and density
def roi_checks_and_output(x_roi, y_roi):
    import numpy as np
    coord_spacing_check = 0.5
    x_roi2 = x_roi.copy()
    y_roi2 = y_roi.copy()
    x_roi2.sort()
    y_roi2.sort()
    unique_spacing_x = np.unique(x_roi2[1:]-x_roi2[0:-1])[0:2]
    unique_spacing_y = np.unique(y_roi2[1:]-y_roi2[0:-1])[0:2]
    expected_unique_spacing = [0,coord_spacing_check]
    if (not (unique_spacing_x==expected_unique_spacing).all()) or (not (unique_spacing_y==expected_unique_spacing).all()): # checks that coord_spacing is coord_spacing_check
        print('ERROR: Coordinate spacing is not', coord_spacing_check)
        exit()
    print('x range:', [x_roi2.min(), x_roi2.max()])
    print('y range:', [y_roi2.min(), y_roi2.max()])
    ncells = len(x_roi2)
    area = (x_roi2.max()-x_roi2.min()) * (y_roi2.max()-y_roi2.min())
    print('[ncells, area, density]:', [ncells, area, ncells/area])
    return(area, unique_spacing_x[1])


# Scale down the text on a set of axes
def scale_text_in_plot(ax, myfontscale):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(myfontscale*item.get_fontsize())


# Write a <TD> HTML element containing an image and the image title
def write_html(f, uroi, title, filepath, imethod=None):
    import os
    f.write('<td align="center">\n')
    if imethod is not None:
        f.write('<b>{}<br>Null method {}<br>{}</b>\n'.format(uroi, imethod, title))
    else:
        f.write('<b>{}<br>{}</b>\n'.format(uroi, title))
    f.write('<br><img src="'+os.path.basename(filepath)+'">\n')
    f.write('</td>\n')
