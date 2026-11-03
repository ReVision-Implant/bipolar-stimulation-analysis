import numpy as np
import matplotlib.pyplot as plt
import tifffile
import pickle
import os
from scipy.stats import wilcoxon, pearsonr, spearmanr, mannwhitneyu, sem, t
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from matplotlib.ticker import FuncFormatter
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

class ExtractFluorescence():
    '''
    Structure of the cell_data from a recording: dictionary with all the cells: cell_data[1] for cell 1, etc
    Structure of the cell_data[x]: dictionary with all the information for this cell:
    'location': (x,y) of the center of the cell
    'intensities': time traces of the intensity for each pattern: [[pattern 1],[pattern 2], etc]. The first time sample is the sample right after stimulation
    'peaks': intensity peaks for each pattern: [pattern 1, pattern 2, etc]
    'active': mask that determines whether a cell is active for a certain pattern (1=active, 0=inactive): [pattern 1, pattern 2, etc]
    'std': standard deviation of the pre-stimulation
    '''
    def __init__(self, directory, recording_name, nb_patterns):
        self.cell_data = None
        self.nb_patterns = nb_patterns
        self.cells_to_pop = [0]

        processed_fn = "cell_data_" + recording_name
        if os.path.exists(directory + "/" + processed_fn):
            print("Loading in existing data for recording " + recording_name)
            self.read_cell_data(directory, processed_fn)
        else:
            print("Loading in new data for recording " + recording_name)
            self.load_neurons(directory=directory, filename = recording_name + "_stimall_seg.npy") # Load in the masks for the neurons
            self.load_recording(directory=directory, filename = recording_name + "_dff.tif")
            self.get_cell_intensities()
            self.split_cell_intensities()
            self.get_cell_peaks()
            self.get_cell_std()
            self.get_cell_centroids()
            self.write_cell_data(directory, processed_fn)
            print("Finished writing new data for recording " + recording_name)

    def load_neurons(self, directory, filename):
        self.neuron_fp = directory + "/" + filename
        try:
            array = np.load(self.neuron_fp, allow_pickle=True).item()
            self.masks = array["masks"]
        except FileNotFoundError:
            print("File not found at the specified path.")

        # Extract unique cell values from masks
        self.unique_cells = set(cell for row in self.masks for cell in row)
        self.nb_cells = len(self.unique_cells) - 1 # "0" is also seen as a cell but this is just the background without any cells

    def load_recording(self, directory, filename):
        self.recording_fp = directory + "/" + filename
        with tifffile.TiffFile(self.recording_fp) as tif:
            # Assuming the image stack is the first series in the TIFF file
            self.recording = tif.series[0].asarray()
            self.recording_length = self.recording.shape[0] # shape = (time, height, width)
        self.nb_pattern_timepoints = round(self.recording_length/self.nb_patterns) # 12 for a downsampled recording, 72 for a non-downsampled recording
        if self.nb_pattern_timepoints != 72:
            print('Possibly something wrong with the number of patterns. Should be : ' + str(self.recording_length/self.nb_pattern_timepoints))

    def get_cell_intensities(self):
        # Initialize a dictionary to hold lists for each cell
        self.cell_data = {cell_value: {'intensities': []} for cell_value in self.unique_cells}

        # Loop over each time point
        for time_point in range(self.recording_length):
            # Initialize or reset sum of intensities and counts for this time point
            cell_sums = {cell_value: 0 for cell_value in self.unique_cells}
            cell_counts = {cell_value: 0 for cell_value in self.unique_cells}

            # Loop over each pixel and update sums and counts
            for y, row in enumerate(self.masks):
                for x, cell_value in enumerate(row):
                    if cell_value > 0: # Skip cell "0", this is just the background
                        intensity = self.recording[time_point, y, x]
                        if intensity >= 0: # NaN and negative values are excluded; such cells are removed below
                            cell_sums[cell_value] += intensity
                            cell_counts[cell_value] += 1
                        else:
                            if cell_value not in self.cells_to_pop:
                                self.cells_to_pop.append(cell_value)
                                print('Added cell ' + str(cell_value) + ' to cells_to_pop because of NaN or negative values.')

            # Compute and store the average for each cell at this time point
            for cell_value in self.unique_cells:
                if cell_counts[cell_value] > 0:
                    avg_intensity = cell_sums[cell_value] / cell_counts[cell_value]
                    self.cell_data[cell_value]['intensities'].append(avg_intensity)
                else:
                    self.cell_data[cell_value]['intensities'].append(None)
                    if cell_value > 0:
                        print("No intensities found for cell " + str(cell_value))

        for cell in self.cells_to_pop:
            self.cell_data.pop(cell) # remove the background values from the dictionary

    def split_cell_intensities(self):
        for cell in self.cell_data.keys():
            intensities = [[]]*self.nb_patterns
            for pattern in range(self.nb_patterns):
                intensities[pattern] = self.cell_data[cell]['intensities'][pattern*self.nb_pattern_timepoints:(pattern+1)*self.nb_pattern_timepoints]
            self.cell_data[cell]['intensities'] = intensities

    def get_cell_peaks(self):
        for cell in self.cell_data.keys():
            peaks = []
            peak_locations = []
            for pattern in range(self.nb_patterns):
                pattern_peak = max(self.cell_data[cell]['intensities'][pattern][:int(self.nb_pattern_timepoints/12*1.6)]) # Look for the maximum in the first second after stimulation (equivalent to 1.6 samples for downsampled data)
                pattern_peak_loc = self.cell_data[cell]['intensities'][pattern].index(pattern_peak)
                peaks.append(pattern_peak)
                peak_locations.append(pattern_peak_loc)
            self.cell_data[cell]['peaks'] = peaks
            self.cell_data[cell]['peak_locations'] = peak_locations

    def get_cell_std(self):
        for cell in self.cell_data.keys():
            background = []
            for pattern in range(self.nb_patterns):
                background.extend(self.cell_data[cell]['intensities'][pattern][-int(self.nb_pattern_timepoints/12*5):]) # Baseline: last 5/12 of each pattern window, i.e. the period before the next stimulation
            mean = np.mean(background)
            std = np.std(background)
            active = []
            for pattern in range(self.nb_patterns):
                if self.cell_data[cell]['peaks'][pattern] > mean + 3*std:
                    active.append(1)
                else:
                    active.append(0)
            self.cell_data[cell]['mean'] = mean
            self.cell_data[cell]['std'] = std
            self.cell_data[cell]['active'] = active

    def get_cell_centroids(self):
        # All centroids are in pixels, not in um
        if self.cell_data == None:
            print("Warning: computing cell centroids without cell intensities.")
            self.cell_data = {cell_value: {'centroid': []} for cell_value in self.unique_cells}
            for cell in self.cells_to_pop:
                self.cell_data.pop(cell) # remove the background values from the dictionary
        else:
            for cell in self.cell_data.keys():
                self.cell_data[cell]['centroid'] = []

        cell_coordinates = {}
        for y, row in enumerate(self.masks):
            for x, cell_value in enumerate(row):
                if cell_value not in cell_coordinates:
                    cell_coordinates[cell_value] = {"x_sum": 0, "y_sum": 0, "count": 0}

                # Accumulate the x and y coordinates
                cell_coordinates[cell_value]["x_sum"] += x
                cell_coordinates[cell_value]["y_sum"] += y
                cell_coordinates[cell_value]["count"] += 1

        # Calculate the central coordinates for each cell
        for cell_value, data in cell_coordinates.items():
            if cell_value not in self.cells_to_pop: # Ignore the background
                x_mean = data["x_sum"] / data["count"]
                y_mean = data["y_sum"] / data["count"]
                self.cell_data[cell_value]['centroid'] = [round(x_mean), round(y_mean)]

    def write_cell_data(self, directory, fn):
        with open(directory + "/" + fn, 'wb') as f:
            pickle.dump(self.cell_data, f)

    def read_cell_data(self, directory, fn):
        with open(directory + "/" + fn, "rb") as f:
            self.cell_data = pickle.load(f)
        self.recording_length = len(self.cell_data[next(iter(self.cell_data.keys()))]['intensities'][0])
        self.nb_pattern_timepoints = round(self.recording_length/self.nb_patterns) # 12 for a downsampled recording, 72 for a non-downsampled recording


class ProcessFluorescence():
    def __init__(self, cell_data, electrode_locations):
        self.cell_data = cell_data
        self.nb_patterns = len(self.cell_data[next(iter(self.cell_data.keys()))]['intensities'])
        self.nb_pattern_timepoints = len(self.cell_data[next(iter(self.cell_data.keys()))]['intensities'][0])
        self.get_common_activity()
        self.max_activity = np.max([self.cell_data[cell]['peaks'] for cell in self.cell_data.keys()]) # Only for visualization purposes, alpha has to be between 0 and 1
        self.load_electrode_locations(electrode_locations[0], electrode_locations[1], electrode_locations[2], electrode_locations[3])
        print("Central electrode: " + str(self.central_electrode))
        print("Return electrodes: " + str(self.return_electrodes))

    def load_electrode_locations(self, central_electrode, return_electrodes, return_electrode_column, return_electrode_layer):
        '''
        Load all electrode locations based on the given electrodes.
        '''
        if not len(return_electrodes) == self.nb_patterns:
            print('Give the correct number of return electrodes: ' + str(self.nb_patterns))
            return
        self.central_electrode = central_electrode
        self.return_electrodes = return_electrodes
        self.return_electrode_layer = return_electrode_layer
        self.return_electrode_column = return_electrode_column

    def get_common_activity(self, patterns=None):
        '''
        Compute the activity that is common for all of the patterns in this slice.
        '''
        for cell in self.cell_data.keys():
            self.cell_data[cell]['common_activity'] = min([self.cell_data[cell]['peaks'][pattern] if self.cell_data[cell]['active'][pattern] == 1 else 0 for pattern in range(self.nb_patterns)])

    def temporal_plot_per_cell(self, cell):
        '''
        Make a plot of all temporal responses (for all patterns) for a given cell.
        '''
        plt.figure()
        for pattern in range(len(self.cell_data[cell]['intensities'])):
            if self.cell_data[cell]['active'][pattern] == 1:
                plt.plot(self.cell_data[cell]['intensities'][pattern], label='Direction ' + str(pattern) + ' (active)', linewidth=3)
                # plt.plot(self.cell_data[cell]['peak_locations'][pattern], self.cell_data[cell]['peaks'][pattern], marker='*', label='Peak of ' + str(pattern))
            else:
                plt.plot(self.cell_data[cell]['intensities'][pattern], label='Direction ' + str(pattern) + ' (inactive)', alpha=0.2, linewidth=3)
        plt.hlines(self.cell_data[cell]['mean'] + 3*self.cell_data[cell]['std'], 0, self.nb_pattern_timepoints-1, label='Activity threshold (µ + 3σ)')
        plt.xticks(range(0,self.nb_pattern_timepoints+1,9), [str(ticklabel) for ticklabel in range(0,9)],fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Time (s)',fontsize=20)
        plt.ylabel('Fluorescence (ΔF/F)',fontsize=20)
        plt.legend(fontsize=14)
        plt.title('Temporal activity of cell ' + str(cell),fontsize=20)
        plt.tight_layout()
        # plt.show()

    def correlation(self, pattern1=0, pattern2=1, plot=True):
        '''
        Use the Spearman correlation coefficient to get a correlation of the two neuronal populations.
        A fluorescence array is defined for each pattern, of equal length.
        Each element of the fluorescence array corresponds to a cell that is active for at least 1 of the patterns
        The value of the element is the fluorescence of the cell when active, and 0 when inactive
        '''

        if pattern1 == pattern2:
            print('Choose 2 different patterns')
            return

        # Value 0 is used for non-active cell, for all cells that are active for at least 1 pattern (even one that is not being compared):
        fluorescence1 = [self.cell_data[cell]['peaks'][pattern1] if self.cell_data[cell]['active'][pattern1] == 1 else 0 for cell in self.cell_data.keys() if 1 in [self.cell_data[cell]['active'][patt] for patt in range(self.nb_patterns)]]
        fluorescence2 = [self.cell_data[cell]['peaks'][pattern2] if self.cell_data[cell]['active'][pattern2] == 1 else 0 for cell in self.cell_data.keys() if 1 in [self.cell_data[cell]['active'][patt] for patt in range(self.nb_patterns)]]

        color_value_X = [self.cell_data[cell]['centroid'][0]/397 for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern1] == 1 or self.cell_data[cell]['active'][pattern2] == 1]
        color_value_Y = [self.cell_data[cell]['centroid'][1]/380 for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern1] == 1 or self.cell_data[cell]['active'][pattern2] == 1]

        corr_coeff = spearmanr(fluorescence1, fluorescence2, alternative='two-sided')
        # print('P-value for Wilcoxon signed-rank test for stim patterns ' + str(pattern1) + ' and ' + str(pattern2) + ' is ' + str(round(signed_rank.pvalue,8)))


        if plot == True:
            color_value='blue'

            plt.figure()
            plt.axline(xy1=(0,0), slope=1, linewidth=2, color='blue')
            plt.scatter(fluorescence1, fluorescence2, c=color_value, s=30, cmap='viridis')
            xlim = plt.xlim()
            ylim = plt.ylim()
            mintick = min([xlim[0], ylim[0]])
            maxtick = max([xlim[1], ylim[1]])
            plt.xlim([mintick, maxtick])
            plt.ylim([mintick, maxtick])
            plt.text(mintick + 7/8*mintick/maxtick, mintick+0.02, 'r = ' + str(round(corr_coeff.statistic,2)), fontsize=15)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel('Pattern ' + str(pattern1) + ' (ΔF/F)', fontsize=20)
            plt.ylabel('Pattern ' + str(pattern2) + ' (ΔF/F)', fontsize=20)
            plt.gca().set_aspect('equal')
            plt.title('Correlation of fluorescence', fontsize=20)
            plt.tight_layout()

        return corr_coeff.statistic

    def overlap(self, pattern1=0, pattern2=1, plot=True):
        '''
        Use the binary pairwise overlap as index of separability between the two neuronal populations.
        '''

        if pattern1 == pattern2:
            print('Choose 2 different patterns')
            return

        # 1 if active, 0 if inactive but active for other pattern, ignore if inactive for both
        fluorescence1 = [1 if self.cell_data[cell]['active'][pattern1] == 1 else 0 for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern1] == 1 or (self.cell_data[cell]['active'][pattern1] == 0 and self.cell_data[cell]['active'][pattern2] == 1)]
        fluorescence2 = [1 if self.cell_data[cell]['active'][pattern2] == 1 else 0 for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern2] == 1 or (self.cell_data[cell]['active'][pattern2] == 0 and self.cell_data[cell]['active'][pattern1] == 1)]
        overlap = [1 if (fluorescence1[i] == 1 and fluorescence2[i] == 1) else 0 for i in range(len(fluorescence1))]

        active_neurons_total = len(fluorescence1)
        active_neurons_1 = fluorescence1.count(1)
        active_neurons_2 = fluorescence2.count(1)
        active_neurons_overlap = overlap.count(1)

        if active_neurons_total > 0:
            return np.round(active_neurons_overlap/active_neurons_total,2), active_neurons_total
        else:
            return 0, 0

    def correlation_per_angle(self, patterns=[0,1,2]):
        '''
        Compute the correlation and overlap for each pairwise combination of patterns, and the angle between the vectors of those 2 patterns.
        '''
        angles = []
        correlations = []
        overlaps = []
        for i in range(len(patterns)):
            for j in range(i+1,len(patterns)):
                angles.append(self.get_electrode_angles(self.central_electrode, self.return_electrodes[patterns[i]], self.return_electrodes[patterns[j]]))
                correlations.append(self.correlation(pattern1=patterns[i], pattern2=patterns[j], plot=False))
                overlaps.append(self.overlap(pattern1=patterns[i], pattern2=patterns[j]))

        angles, correlations, overlaps = zip(*sorted(zip(angles, correlations, overlaps))) # Sort the correlations in ascending order
        return angles, correlations, overlaps

    @staticmethod
    def get_electrode_angles(central, elec1, elec2):
        '''
        Compute the angle between the vectors of 2 patterns, from a given central electrode and 2 given return electrodes.
        '''
        P12 = np.sqrt((central[0]-elec1[0])**2 + (central[1]-elec1[1])**2)
        P13 = np.sqrt((central[0]-elec2[0])**2 + (central[1]-elec2[1])**2)
        P23 = np.sqrt((elec1[0]-elec2[0])**2 + (elec1[1]-elec2[1])**2)
        return np.rad2deg(np.arccos((P12**2 + P13**2 - P23**2)/(2*P12*P13)))

    def plot_activity(self, pattern=0, neurons='all', arrow=False):
        '''
        Make a plot of the spatial activity of all cells for a given pattern.
        '''
        plt.figure()
        for cell in self.cell_data.keys():
            if neurons == 'all':
                cell_intensity = self.cell_data[cell]['peaks'][pattern] if self.cell_data[cell]['active'][pattern] == True else 0
            elif neurons == 'unique':
                cell_intensity = self.cell_data[cell]['peaks'][pattern] - self.cell_data[cell]['common_activity'] if self.cell_data[cell]['active'][pattern] == True else 0
            elif neurons == 'common':
                cell_intensity = self.cell_data[cell]['common_activity']
            plt.scatter(self.cell_data[cell]['centroid'][0], self.cell_data[cell]['centroid'][1], s=90, c="blue", alpha=cell_intensity/self.max_activity)

        # Scaling factor to go from pixels to µm
        scaling_factor = 1.27

        # Define the new tick labels and their scaled positions
        tick_labels_x = [0, 100, 200, 300, 400, 500]  # Desired tick labels
        scaled_positions_x = [label / scaling_factor for label in tick_labels_x]  # Scale their positions
        tick_labels_y = [0, 100, 200, 300, 400]  # Desired tick labels
        scaled_positions_y = [label / scaling_factor for label in tick_labels_y]  # Scale their positions

        # Set the new ticks and labels
        plt.xticks(ticks=scaled_positions_x, labels=tick_labels_x, fontsize=15)
        plt.yticks(ticks=scaled_positions_y, labels=tick_labels_y, fontsize=15)

        plt.xlabel('X Coordinate [µm]', fontsize=20)
        plt.ylabel('Y Coordinate [µm]', fontsize=20)
        plt.xlim([0, 397])
        plt.ylim([0, 380])
        plt.gca().invert_yaxis()  # Invert y-axis for better comparison with ImageJ
        plt.gca().set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal
        if arrow==True:
            plt.arrow(self.central_electrode[0], self.central_electrode[1], self.return_electrodes[pattern][0]-self.central_electrode[0], self.return_electrodes[pattern][1]-self.central_electrode[1], length_includes_head=True, width=5, head_width=20, color='black')
        plt.title('Active neurons (' + neurons + ') for direction ' + str(pattern), fontsize=20)
        plt.tight_layout()
        # plt.show()

    def pca(self):
        '''
        Perform Principal Component Analysis on all patterns and cells in a certain slice.
        '''
        activity_matrix = [] # N x 13 with N the number of ROIs and 13 the number of current directions
        for pattern in range(self.nb_patterns):
            # Take all fluorescence values per stimulation pattern, but only for cells that are active for at least 1 pattern
            activity_matrix.append([self.cell_data[cell]['peaks'][pattern] for cell in self.cell_data.keys() if 1 in [self.cell_data[cell]['active'][patt] for patt in range(self.nb_patterns)]])
        activity_matrix = np.array(activity_matrix).T # shape = N x 13
        if activity_matrix.shape[0] <= activity_matrix.shape[1]: # PCA requires more active cells than patterns
            return None
        pca = PCA(n_components=13) # Apply PCA with 13 principal components
        pca.fit(activity_matrix)

        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        return cumulative_variance

    @staticmethod
    def gauss(x, x0, sigma, amp, offset):
        '''
        Create a Gaussian function.
        '''
        return offset + amp * np.exp(-(x-x0)**2/(2*sigma**2))

    def spatial_analysis_polarity(self, pattern=0, first_electrode=None, second_electrode=(100,100), plot=True):
        '''
        Compute the spatial characteristics (centroid) for cathodic-first vs anodic-first stimulation.
        '''
        if first_electrode==None:
            first_electrode=self.central_electrode
        coordinates = np.array([self.cell_data[cell]['centroid'] for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern] == 1]) # Only take active cells into account
        fluorescence = np.array([self.cell_data[cell]['peaks'][pattern] for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern] == 1])
        if len(coordinates) == 0:
            print('No active neurons so no spatial analysis possible')
            return 100 # No active neurons

        projected_coordinates, fluorescence = self.project_neurons(coordinates, fluorescence, point1=first_electrode, point2=second_electrode, plot=plot)
        centroid, stdev = self.fit_neurons_stdev(projected_coordinates, fluorescence, plot=plot) # Weighted mean of the projected positions (no Gaussian fit)
        return centroid

    def spatial_analysis(self, pattern=0, fitting_algorithm='gaussian', plot=True):
        '''
        Compute the spatial characteristics (centroids and standard deviations) for a given pattern.
        It first projects the active neurons on a certain axis (defined by the return electrodes).
        Then, these projected neurons are estimated using a Gaussian estimate, or a simple "centroid"-fit.
        '''

        coordinates = np.array([self.cell_data[cell]['centroid'] for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern] == 1]) # # Only take active cells into account
        fluorescence = np.array([self.cell_data[cell]['peaks'][pattern] for cell in self.cell_data.keys() if self.cell_data[cell]['active'][pattern] == 1])

        if len(coordinates) == 0:
            print('No active neurons so no spatial analysis possible')
            return None, None, None, None

        # Centroid along cortical column:
        projected_coordinates, fluorescence = self.project_neurons(coordinates, fluorescence, point1=self.central_electrode, point2=self.return_electrode_column, plot=False)
        centroid_along_column = self.fit_neurons_kde(projected_coordinates, fluorescence, plot=False)
        centroid_along_column_2d = (self.central_electrode[0] + centroid_along_column*(self.return_electrode_column[0]-self.central_electrode[0]), self.central_electrode[1] + centroid_along_column*(self.return_electrode_column[1]-self.central_electrode[1])) # Compute the X and Y for this maximal value

        # Centroid along cortical layer:

        # Project 2D neurons on a certain 1D axis
        projected_coordinates, fluorescence = self.project_neurons(coordinates, fluorescence, point1=self.central_electrode, point2=self.return_electrode_layer, plot=plot)
        # Fit projected neurons
        if fitting_algorithm == 'gaussian':
            if self.get_nb_active_neurons(pattern=pattern)[0] < 20:
                print('Attention: only ' + str(self.get_nb_active_neurons(pattern=pattern)) + ' active neurons in pattern ' + str(pattern) + ', too low for Gaussian fit. Changing to stdev.')
                centroid_along_layer, stdev_along_layer = self.fit_neurons_stdev(projected_coordinates, fluorescence, plot=plot)
            else:
                [centroid_along_layer, stdev_along_layer, gaussian_amp, gaussian_offset] = self.fit_neurons_gaussian(projected_coordinates, fluorescence, plot=plot)
        elif fitting_algorithm == 'kde':
            centroid_along_layer = self.fit_neurons_kde(projected_coordinates, fluorescence, plot=plot)
            stdev_along_layer = None
        elif fitting_algorithm == 'stdev':
            centroid_along_layer, stdev_along_layer = self.fit_neurons_stdev(projected_coordinates, fluorescence, plot=plot)
        else:
            print('Choose an existing fitting algorithm: gaussian/kde/stdev')
            return

        centroid_along_layer_2d = (self.central_electrode[0] + centroid_along_layer*(self.return_electrode_layer[0]-self.central_electrode[0]), self.central_electrode[1] + centroid_along_layer*(self.return_electrode_layer[1]-self.central_electrode[1])) # Compute the X and Y for this maximal value
        # Project center on projection axis back to 2D to get 2D-centroid
        centroid_2d = self.intersect_projections(centroid_along_column_2d, centroid_along_layer_2d)

        return centroid_along_column, centroid_along_layer, stdev_along_layer, centroid_2d

    def project_neurons(self, original_points, amplitude, point1=(0,0), point2=(397,380), plot=False):
        '''
        Project the neurons in a 2D-space into a 1D-space, given projection points that define a line in the 2D-space.
        '''
        direction_vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
        projected_points = np.array([np.dot(np.array(centroid) - np.array(point1), direction_vector) / np.dot(direction_vector, direction_vector) for centroid in original_points])

        if len(original_points) == 0:
            print('No active neurons so no projection possible.')
            return

        projected_points, amplitude = zip(*sorted(zip(projected_points, amplitude))) # Sort the projected_points in ascending order
        projected_points = np.array(projected_points)
        amplitude = np.array(amplitude)

        if plot==True:
            fig, ax = plt.subplots(2, 1)
            ax = ax.flatten()

            # Plot for original points, line, and projected points
            ax[0].plot([point1[0], point2[0]], [point1[1], point2[1]], color='blue')
            for cell in range(len(original_points)):
                projected_coordinate = [point1[0] + projected_points[cell] * direction_vector[0], point1[1] + projected_points[cell] * direction_vector[1]]
                ax[0].scatter(original_points[cell][0], original_points[cell][1], s=30, c="blue", alpha=amplitude[cell]/self.max_activity, label='Original')
                ax[0].scatter(projected_coordinate[0], projected_coordinate[1], s=15, c="red", alpha=amplitude[cell]/self.max_activity, label='Projected')

            ax[0].set_xlabel('X Coordinate')
            ax[0].set_ylabel('Y Coordinate')
            ax[0].set_xlim([0, 397])
            ax[0].set_ylim([0, 380])
            ax[0].invert_yaxis()  # Invert y-axis for better comparison with ImageJ
            ax[0].set_aspect('equal', adjustable='box')  # Set aspect ratio to be equal
            ax[0].set_title('Projection of points onto projection axis')
            handles, labels = ax[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles)) # To remove duplicate labels because of the for loop
            ax[0].legend(by_label.values(), by_label.keys())

            # Plot for points along the projection axis
            ax[1].scatter(projected_points, amplitude, color='blue')
            ax[1].set_xticks([])
            ax[1].set_ylabel('Amplitude')
            ax[1].set_title('Points along projection axis')
            ax[1].set_aspect('auto')

            fig.tight_layout()

        return projected_points, amplitude

    def fit_neurons_gaussian(self, points, amplitude, plot=False):
        '''
        Perform a Gaussian fit on an array of (projected 1D) neurons.
        '''
        x0_estimated, mu0_estimated = self.fit_neurons_stdev(points, amplitude, plot=False) # Take the mean and standard deviation as initial guess for the peak and standard deviation of the Gaussian

        nb_bins = 20
        # Create nb_bins equidistant bins
        bins = np.linspace(points[0], points[-1], nb_bins+1)  # nb_bins + 1 edges
        cumulative_values = np.zeros(nb_bins)

        bin_index = 0
        # Accumulate the fluorescence values in the bins
        for i in range(len(points)):
            # Check if the projected point exceeds the current bin edge
            while bin_index < nb_bins-1 and points[i] > bins[bin_index + 1]:
                bin_index += 1
            # Add the fluorescence value to the corresponding bin
            cumulative_values[bin_index] += amplitude[i]

        # Prepare the new list with mid-points of the bins and their corresponding cumulative values
        points = (bins[:-1] + bins[1:]) / 2
        amplitude = cumulative_values


        p0 = [x0_estimated, mu0_estimated, max(amplitude), min(amplitude)] # Initial guess for p = [mu, sigma, amplitude, offset]
        p_bounds = ((x0_estimated*0.8 if x0_estimated >= 0 else x0_estimated*1.2, 0, min(amplitude), min(amplitude)*0.5), (x0_estimated*1.2 if x0_estimated >= 0 else x0_estimated*0.8, np.inf, max(amplitude)*2, max(amplitude))) # Boundaries for the different parameters
        popt, pcov = curve_fit(self.gauss, points, amplitude, p0 =p0, bounds=p_bounds, maxfev=5000, method='trf', loss='linear') # Fit the points and their values to the given function. It's possible to choose a different loss function for regularization against outliers (eg cauchy)
        fitted_curve = self.gauss(points, popt[0], popt[1], popt[2], popt[3]) # Compute the fitted curve for the projected_points

        if plot==True:
            plt.figure()
            # Plot for points along the projection axis
            plt.scatter(points, amplitude, color='blue')
            plt.plot(points, fitted_curve, color='red', linestyle='-')
            plt.xlabel('Density along projected axis')
            plt.xticks([0,1],["Point 1", "Point 2"])
            plt.title('1D Gaussian fit of projected points')

        return popt

    def fit_neurons_kde(self, points, amplitude, plot=False):
        '''
        Perform a Kernel Density Estimate on an array of (projected 1D) neurons.
        '''
        # Perform kernel density estimation
        kde = KernelDensity(bandwidth=0.6, kernel='gaussian') # Bandwidth in projected units (0 = first electrode, 1 = second electrode)
        kde.fit(points.reshape(-1, 1), sample_weight=amplitude) # Train model

        # Define grid along the projected axis
        grid_size = 100
        grid = np.linspace(min(points), max(points), grid_size).reshape(-1, 1)
        density = np.exp(kde.score_samples(grid))
        centroid = grid[np.argmax(density)][0]

        if plot==True:
            plt.figure()
            # Plot for points along the projection axis
            plt.scatter(points, amplitude, color='blue')
            plt.plot(grid, density, color='red', linestyle='-')
            plt.scatter(centroid, 0, color='red', linewidth=10)
            plt.xlabel('Density along projected axis')
            plt.xticks([0,1],["Point 1", "Point 2"])
            plt.title('1D KDE of projected points')

        return centroid


    def fit_neurons_stdev(self, points, amplitude, plot=False):
        '''
        Perform a fit on an array of (projected 1D) neurons, by computing the weighted mean and standard deviation.
        Takes the fluorescence amplitude into account when computing the mean.
        '''
        centroid = np.average(points, weights=amplitude)
        stdev = np.std(points)

        if plot==True:
            plt.figure()
            # Plot for points along the projection axis
            plt.scatter(points, amplitude, color='blue')
            plt.scatter(centroid, 0, color='red')
            plt.plot([centroid-stdev, centroid+stdev], [0, 0], color='red')
            plt.xlabel('Density along projected axis')
            plt.xticks([0,1],["Point 1", "Point 2"])
            plt.title('Centroid and standard deviation of projected points')

        return centroid, stdev

    def intersect_projections(self, centroid_column_xy, centroid_layer_xy):
        '''
        Compute the intersection of 2 1D-lines back into the 2D-space.
        '''
        # Line along the layer through the column centroid: a1*y+b1*x+c1=0
        a1 = self.return_electrode_layer[1]-self.central_electrode[1]
        b1 = -(self.return_electrode_layer[0]-self.central_electrode[0])
        c1 = -b1*centroid_column_xy[1] - a1*centroid_column_xy[0]

        # Line along the column through the layer centroid: a2*y+b2*x+c2=0
        a2 = self.return_electrode_column[1]-self.central_electrode[1]
        b2 = -(self.return_electrode_column[0]-self.central_electrode[0])
        c2 = -b2*centroid_layer_xy[1] - a2*centroid_layer_xy[0]

        max_xy = ((b1*c2-b2*c1)/(a1*b2-a2*b1), (c1*a2-c2*a1)/(a1*b2-a2*b1))

        return max_xy

    def get_nb_active_neurons(self, pattern=0):
        '''
        Compute the number of active neurons in this slice for a given pattern.
        '''
        nb_active_neurons = 0
        cumulative_activity = 0
        for cell in self.cell_data.keys():
            if self.cell_data[cell]['active'][pattern] == 1:
                nb_active_neurons += 1
                cumulative_activity += self.cell_data[cell]['peaks'][pattern]
        return nb_active_neurons, cumulative_activity


    def amplitude_cumulative_activity(self, amp_range = 'all', numbering = 'old', plot=True):
        '''
        Compute the activity for increasing amplitude, using the number of neurons as metric.
        Function works on both the old and new way of numbering amplitude stimulation (old = with asymmetric pulses as well; new = without).
        Function works when the full range of amplitudes is measured, or only half of the range.
        '''
        all_amplitudes = [6.6, 12.5, 15.4, 22.7, 25.6, 28.6, 31.5, 34.4]
        if amp_range == 'all':
            amplitudes = all_amplitudes
        elif amp_range == 'low':
            amplitudes = all_amplitudes[:4]
        elif amp_range == 'high':
            amplitudes = all_amplitudes[-4:]

        if numbering == 'old':
            step = 2
        elif numbering == 'new':
            step = 1
        else:
            print('Choose old or new as numbering term.')
            return
        nb_active_neurons = [0]*int(self.nb_patterns/step)
        cumulative_activity = [0]*int(self.nb_patterns/step)
        for cell in self.cell_data.keys():
            for pattern in range(len(nb_active_neurons)):
                if self.cell_data[cell]['active'][pattern*step] == 1:
                    nb_active_neurons[pattern] += 1
                    cumulative_activity[pattern] += self.cell_data[cell]['peaks'][pattern*step]


        if plot==True:
            fig, ax1 = plt.subplots()
            ax1.plot(amplitudes, nb_active_neurons, color='blue', linewidth=3)
            ax1.set_ylabel('Nb. of active neurons', fontsize=20)
            ax1.set_xlabel('Current amplitude [µA]', fontsize=20)
            ax1.tick_params(axis='both', labelsize=15)
            ax1.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
            ax1.tick_params(axis='x', labelsize=15)
            plt.title('Activity for increasing stimulation amplitude', fontsize=20)
            plt.tight_layout()

        return amplitudes, nb_active_neurons, cumulative_activity

def get_all_electrode_locations(central, return_col, return_layer):
    '''
    Computes the electrode locations of all return electrodes based on the given central electrode and 2 given return electrodes, using the rectangular grid.
    Assumes a rectangular grid, always take the middle of the shank
    Procedure to determine the central electrode, return electrode in column (to the bottom left) and return electrode in layer (to the bottom right)
    1) Open in ImageJ the '_baseline.tif' of one of the original recordings (electrodes are not clearly visible enough in the dff-files)
    2) For all three electrodes, take the middle of the shank as electrode location. Use the pixels (number between brackets)
    3) Check: (x1, y1), (x2, y2), (x3, y3): x3 > x1 > x2
    '''
    all_electrodes = [return_col, return_layer]
    difference_row = tuple(np.subtract(return_layer, central))
    all_electrodes.append(add_tuple(return_col, difference_row, 1/2))
    all_electrodes.append(add_tuple(central, difference_row, 2))
    all_electrodes.append(add_tuple(return_col, difference_row, 2))
    all_electrodes.append(add_tuple(return_col, difference_row, 3/2))
    all_electrodes.append(add_tuple(return_col, difference_row, 1))
    all_electrodes.append(add_tuple(return_col, difference_row, -1/2))
    all_electrodes.append(add_tuple(return_col, difference_row, -1))
    all_electrodes.append(add_tuple(return_col, difference_row, -3/2))
    all_electrodes.append(add_tuple(return_col, difference_row, -2))
    all_electrodes.append(add_tuple(central, difference_row, -2))
    all_electrodes.append(add_tuple(central, difference_row, -1))
    print('All electrodes: ', all_electrodes)
    return all_electrodes

def add_tuple(tuple1, tuple2, multiplication):
    '''
    Adds the elements of 2 tuples.
    '''
    return (int(tuple1[0] + multiplication*tuple2[0]), int(tuple1[1] + multiplication*tuple2[1]))

if __name__ == "__main__":

    ############################################################
    ##################### DATA LOADING #########################
    ############################################################

    # The analysis below expects one ProcessFluorescence object per dataset (dF/F stack + cell masks), appended to the lists below.
    # The list indices used in the analysis assume the datasets are appended in the same order as for the paper.
    # Each dataset directory must contain '<name>_stimall_seg.npy' (cell masks) and '<name>_dff.tif' (dF/F stack).
    # Electrode locations (in pixels): [central_electrode, return_electrodes, return_electrode_column, return_electrode_layer]
    # Example for one dataset with 13 stimulation directions:
    #
    # directory = "path/to/dataset_directory"
    # experiment = ExtractFluorescence(directory=directory, recording_name="name", nb_patterns=13)
    # central, column, layer = (x0, y0), (x1, y1), (x2, y2)
    # electrode_locations = [central, get_all_electrode_locations(central, column, layer), column, layer]
    # list_directions20.append(ProcessFluorescence(experiment.cell_data, electrode_locations))

    list_directions20 = [] # Directions, 20 uA (13 patterns)
    list_directions10 = [] # Directions, 10 uA (13 patterns)
    list_asymm = [] # Polarity and asymmetry (24 patterns)
    list_variability = [] # Biological variability (48 patterns)
    list_depth = [] # Depth (8 patterns)
    list_local = [] # Hyper-local (6 patterns)
    list_amplitude = [] # Amplitude (8 or 16 patterns)

    if not all([list_directions20, list_directions10, list_asymm, list_variability, list_depth, list_local, list_amplitude]):
        raise SystemExit("No data loaded: add the datasets to the lists in the DATA LOADING section before running the analysis.")

    ############################################################
    ######### DATA ANALYSIS DIRECTIONALITY SPATIAL #############
    ############################################################

    centroids = [[], [], [], [], [], [], [], [], [], [], [], [], []]

    for slice in list_directions20:
        for pattern in range(0,slice.nb_patterns):
            nb_active_neurons = slice.get_nb_active_neurons(pattern=pattern)[0]
            if nb_active_neurons < 30:
                continue
            centroid_column, centroid_layer, stdev, _, = slice.spatial_analysis(pattern=pattern, fitting_algorithm='gaussian', plot=False)
            centroids[pattern].append([centroid_column, centroid_layer, stdev])

    colors = ['red', 'blue', 'orange', 'purple', 'yellow', 'purple', 'grey', 'green']
    return_elec = [(0, 1), (1, 0), (0.5, 1), (2, 0), (2, 1), (1.5, 1), (1, 1), (-0.5, 1), (-1, 1)]

    plt.figure()
    for pattern in [0, 1, 2, 6, 7]:
        centroid_collection = []
        for centroid in range(len(centroids[pattern])):
            plt.plot(centroids[pattern][centroid][1], centroids[pattern][centroid][0], 'o', markersize=5, color=colors[pattern], alpha=0.5, label='Intra-slice centroids')
            # Plot a polygon for illustration purposes
            centroid_collection.append([centroids[pattern][centroid][1], centroids[pattern][centroid][0]])

        plt.plot(np.mean([centroids[pattern][centroid][1] for centroid in range(len(centroids[pattern]))]), np.mean([centroids[pattern][centroid][0] for centroid in range(len(centroids[pattern]))]), 'o', markersize=10, color=colors[pattern], label='Inter-slice centroid', zorder=3)
        plt.plot([np.mean([centroids[pattern][centroid][1] for centroid in range(len(centroids[pattern]))]) - np.mean([centroids[pattern][centroid][2] for centroid in range(len(centroids[pattern]))]), \
                                                                        np.mean([centroids[pattern][centroid][1] for centroid in range(len(centroids[pattern]))]) + np.mean([centroids[pattern][centroid][2] for centroid in range(len(centroids[pattern]))])],
                                                                        2*[np.mean([centroids[pattern][centroid][0] for centroid in range(len(centroids[pattern]))])], '-', linewidth=3, color=colors[pattern], label='Inter-slice centroid', zorder=3)
        plt.arrow(0, 0, return_elec[pattern][0], return_elec[pattern][1], length_includes_head=True, width=0.01, head_width=0.05, color=colors[pattern])

        # Plot a polygon for illustration purposes
        hull = ConvexHull(np.array(centroid_collection))
        for simplex in hull.simplices:
            plt.plot([centroid_collection[simplex[0]][0], centroid_collection[simplex[1]][0]], [centroid_collection[simplex[0]][1], centroid_collection[simplex[1]][1]], color=colors[pattern], alpha=0.6)
        plt.fill([centroid_collection[vertix][0] for vertix in hull.vertices], [centroid_collection[vertix][1] for vertix in hull.vertices], color=colors[pattern], alpha=0.2)

    plt.xlabel('Distance parallel to surface [µm]', fontsize=20)
    plt.ylabel('Distance perpendicular \nto surface [µm]', fontsize=20)
    plt.xticks(ticks=[-1, -0.66, -0.33, 0, 0.33, 0.66, 1], labels=['-150', '-100', '-50', '0', '50', '100', '150'], fontsize=15)
    plt.yticks(ticks=[-0.33, 0, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33], labels=['-50', '0', '50', '100', '150', '200', '250', '300', '350'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)
    plt.gca().get_legend().legend_handles[0].set_color('black')
    plt.gca().get_legend().legend_handles[1].set_color('black')

    plt.text(-0.95, 0.15, 'Layer 2/3', fontsize=15)
    plt.hlines(0.05, -0.95, -0.25, colors='black', linestyles='dashed')
    plt.text(-0.95, -0.15, 'Layer 4', fontsize=15)

    plt.title('Activity centroids for different directions', fontsize=20)
    plt.tight_layout()


    ############################################################
    ######### DATA ANALYSIS BIOLOGICAL VARIABILITY #############
    ############################################################

    correlations = []

    for slice in list_variability:
        for pattern in range(0,slice.nb_patterns,2):
            correlations.append([slice.correlation(pattern1=pattern, pattern2=int(pattern+1), plot=False), slice.overlap(pattern1=pattern, pattern2=int(pattern+1), plot=False)])

    variability_correlations = [comparison[0] for comparison in correlations if comparison[1][1] > 50] # Only comparisons with more than 50 neurons active in either pattern
    variability_overlaps = [comparison[1][0] for comparison in correlations if comparison[1][1] > 50] # Only comparisons with more than 50 neurons active in either pattern

    variability_correlations_boundaries = [np.mean(variability_correlations) - np.std(variability_correlations), np.mean(variability_correlations) + np.std(variability_correlations)]
    variability_overlaps_boundaries = [np.mean(variability_overlaps) - np.std(variability_overlaps), np.mean(variability_overlaps) + np.std(variability_overlaps)]

    print("Number of samples for variability: " + str(len(variability_correlations)))
    print("Mean +- std for variability correlations: " + str(variability_correlations_boundaries))
    print("Mean +- std for variability overlaps: " + str(variability_overlaps_boundaries))

    ############################################################
    ######### DATA ANALYSIS ASYMMETRY AND POLARITY #############
    ############################################################

    # 3 different combinations: different polarity, different asymmetry, different polarity and asymmetry

    correlations_polar_20 = []
    correlations_asymm_20 = []
    correlations_both_20 = []
    # Formula: centroid_shift = centroid from anodic-first - centroid from cathodic-first, divided by 2
    # P2: the (virtual) return electrode lies 168 um from the central electrode instead of 150 um, so its shift is multiplied by 168/150 = 1.12

    for slice in list_asymm:
        for direction_pattern in [1, 9, 17]: # cathodic-first and symmetric, 20uA
            if direction_pattern == 1: # P0
                second_electrode = slice.return_electrode_column
                distance_factor = 1
            elif direction_pattern == 9: # P1
                second_electrode = slice.return_electrode_layer
                distance_factor = 1
            elif direction_pattern == 17: # P2
                second_electrode = slice.return_electrodes[2]
                distance_factor = 1.12

            correlations_polar_20.append([slice.correlation(pattern1=direction_pattern+0, pattern2=direction_pattern+2, plot=False), slice.overlap(pattern1=direction_pattern+0, pattern2=direction_pattern+2, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+2, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+0, second_electrode=second_electrode, plot=False))/2])
            correlations_polar_20.append([slice.correlation(pattern1=direction_pattern+4, pattern2=direction_pattern+6, plot=False), slice.overlap(pattern1=direction_pattern+4, pattern2=direction_pattern+6, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+6, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+4, second_electrode=second_electrode, plot=False))/2])

            correlations_asymm_20.append([slice.correlation(pattern1=direction_pattern+0, pattern2=direction_pattern+4, plot=False), slice.overlap(pattern1=direction_pattern+0, pattern2=direction_pattern+4, plot=False)])
            correlations_asymm_20.append([slice.correlation(pattern1=direction_pattern+2, pattern2=direction_pattern+6, plot=False), slice.overlap(pattern1=direction_pattern+2, pattern2=direction_pattern+6, plot=False)])

            correlations_both_20.append([slice.correlation(pattern1=direction_pattern+0, pattern2=direction_pattern+6, plot=False), slice.overlap(pattern1=direction_pattern+0, pattern2=direction_pattern+6, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+6, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+0, second_electrode=second_electrode, plot=False))/2])
            correlations_both_20.append([slice.correlation(pattern1=direction_pattern+2, pattern2=direction_pattern+4, plot=False), slice.overlap(pattern1=direction_pattern+2, pattern2=direction_pattern+4, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+2, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+4, second_electrode=second_electrode, plot=False))/2])

    correlations_polar_10 = []
    correlations_asymm_10 = []
    correlations_both_10 = []

    for slice in list_asymm:
        for direction_pattern in [0, 8, 16]: # cathodic-first and symmetric, 10uA
            if direction_pattern == 0: # P0
                second_electrode = slice.return_electrode_column
                distance_factor = 1
            elif direction_pattern == 8: # P1
                second_electrode = slice.return_electrode_layer
                distance_factor = 1
            elif direction_pattern == 16: # P2
                second_electrode = slice.return_electrodes[2]
                distance_factor = 1.12

            correlations_polar_10.append([slice.correlation(pattern1=direction_pattern+0, pattern2=direction_pattern+2, plot=False), slice.overlap(pattern1=direction_pattern+0, pattern2=direction_pattern+2, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+2, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+0, second_electrode=second_electrode, plot=False))/2])
            correlations_polar_10.append([slice.correlation(pattern1=direction_pattern+4, pattern2=direction_pattern+6, plot=False), slice.overlap(pattern1=direction_pattern+4, pattern2=direction_pattern+6, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+6, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+4, second_electrode=second_electrode, plot=False))/2])

            correlations_asymm_10.append([slice.correlation(pattern1=direction_pattern+0, pattern2=direction_pattern+4, plot=False), slice.overlap(pattern1=direction_pattern+0, pattern2=direction_pattern+4, plot=False)])
            correlations_asymm_10.append([slice.correlation(pattern1=direction_pattern+2, pattern2=direction_pattern+6, plot=False), slice.overlap(pattern1=direction_pattern+2, pattern2=direction_pattern+6, plot=False)])

            correlations_both_10.append([slice.correlation(pattern1=direction_pattern+0, pattern2=direction_pattern+6, plot=False), slice.overlap(pattern1=direction_pattern+0, pattern2=direction_pattern+6, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+6, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+0, second_electrode=second_electrode, plot=False))/2])
            correlations_both_10.append([slice.correlation(pattern1=direction_pattern+2, pattern2=direction_pattern+4, plot=False), slice.overlap(pattern1=direction_pattern+2, pattern2=direction_pattern+4, plot=False),
                                        distance_factor*(slice.spatial_analysis_polarity(pattern=direction_pattern+2, second_electrode=second_electrode, plot=False) - slice.spatial_analysis_polarity(pattern=direction_pattern+4, second_electrode=second_electrode, plot=False))/2])



    threshold = 50

    fig, ax = plt.subplots()
    markersize=5
    colors = ['blue', 'orange']
    color = 'grey'
    alpha = 0.5

    correlations = [correlation[0] for correlation in correlations_polar_20 if correlation[1][1] > threshold]
    for dot in correlations:
        plt.plot(0.9, dot, 'o', markersize=markersize, color=colors[0], alpha=alpha, label='20 µA')
    plt.plot(0.9, np.mean(correlations), 'o', markersize=markersize*2, color=colors[0])
    ax.errorbar(0.9, np.mean(correlations), yerr=np.std(correlations), capsize=20, capthick=2, color=colors[0])

    correlations = [correlation[0] for correlation in correlations_polar_10 if correlation[1][1] > threshold]
    for dot in correlations:
        plt.plot(1.1, dot, 'o', markersize=markersize, color=colors[1], alpha=alpha, label='10 µA')
    plt.plot(1.1, np.mean(correlations), 'o', markersize=markersize*2, color=colors[1])
    ax.errorbar(1.1, np.mean(correlations), yerr=np.std(correlations), capsize=20, capthick=2, color=colors[1])




    correlations = [correlation[0] for correlation in correlations_asymm_20 if correlation[1][1] > threshold]
    for dot in correlations:
        plt.plot(1.9, dot, 'o', markersize=markersize, color=colors[0], alpha=alpha)
    plt.plot(1.9, np.mean(correlations), 'o', markersize=markersize*2, color=colors[0])
    ax.errorbar(1.9, np.mean(correlations), yerr=np.std(correlations), capsize=20, capthick=2, color=colors[0])

    correlations = [correlation[0] for correlation in correlations_asymm_10 if correlation[1][1] > threshold]
    for dot in correlations:
        plt.plot(2.1, dot, 'o', markersize=markersize, color=colors[1], alpha=alpha)
    plt.plot(2.1, np.mean(correlations), 'o', markersize=markersize*2, color=colors[1])
    ax.errorbar(2.1, np.mean(correlations), yerr=np.std(correlations), capsize=20, capthick=2, color=colors[1])




    correlations = [correlation[0] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    for dot in correlations:
        plt.plot(2.9, dot, 'o', markersize=markersize, color=colors[0], alpha=alpha)
    plt.plot(2.9, np.mean(correlations), 'o', markersize=markersize*2, color=colors[0])
    ax.errorbar(2.9, np.mean(correlations), yerr=np.std(correlations), capsize=20, capthick=2, color=colors[0])

    correlations = [correlation[0] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    for dot in correlations:
        plt.plot(3.1, dot, 'o', markersize=markersize, color=colors[1], alpha=alpha)
    plt.plot(3.1, np.mean(correlations), 'o', markersize=markersize*2, color=colors[1])
    ax.errorbar(3.1, np.mean(correlations), yerr=np.std(correlations), capsize=20, capthick=2, color=colors[1])

    plt.axhspan(variability_correlations_boundaries[0], variability_correlations_boundaries[1], facecolor='grey', alpha=0.2)

    plt.ylim([0, 1])
    plt.yticks(fontsize=15)
    plt.ylabel('Spearman correlation coeff.', fontsize=20)
    plt.xticks([1,2,3], ['Polarity', 'Symmetry', 'Both'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # To remove duplicate labels because of the for loop
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Correlation for different parameters', fontsize=20)
    fig.tight_layout()

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (polarity 20 vs 10):")
    dataset1 = [correlation[0] for correlation in correlations_polar_20 if correlation[1][1] > threshold]
    dataset2 = [correlation[0] for correlation in correlations_polar_10 if correlation[1][1] > threshold]
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (asymmetry 20 vs 10):")
    dataset1 = [correlation[0] for correlation in correlations_asymm_20 if correlation[1][1] > threshold]
    dataset2 = [correlation[0] for correlation in correlations_asymm_10 if correlation[1][1] > threshold]
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (both 20 vs 10):")
    dataset1 = [correlation[0] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    dataset2 = [correlation[0] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (polarity 20uA vs biological variability):")
    dataset1 = [correlation[0] for correlation in correlations_polar_20 if correlation[1][1] > threshold]
    dataset2 = variability_correlations
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (polarity 10uA vs biological variability):")
    dataset1 = [correlation[0] for correlation in correlations_polar_10 if correlation[1][1] > threshold]
    dataset2 = variability_correlations
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (asymmetry 20uA vs biological variability):")
    dataset1 = [correlation[0] for correlation in correlations_asymm_20 if correlation[1][1] > threshold]
    dataset2 = variability_correlations
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (asymmetry 10uA vs biological variability):")
    dataset1 = [correlation[0] for correlation in correlations_asymm_10 if correlation[1][1] > threshold]
    dataset2 = variability_correlations
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (both 20uA vs biological variability):")
    dataset1 = [correlation[0] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    dataset2 = variability_correlations
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for correlations (both 10uA vs biological variability):")
    dataset1 = [correlation[0] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    dataset2 = variability_correlations
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")


    fig, ax = plt.subplots()
    markersize=5
    colors = ['blue', 'orange']
    color = 'grey'
    alpha = 0.5

    overlap = [correlation[1][0] for correlation in correlations_polar_20 if correlation[1][1] > threshold]
    for dot in overlap:
        plt.plot(0.9, dot, 'o', markersize=markersize, color=colors[0], alpha=alpha, label='20 µA')
    plt.plot(0.9, np.mean(overlap), 'o', markersize=markersize*2, color=colors[0])
    ax.errorbar(0.9, np.mean(overlap), yerr=np.std(overlap), capsize=20, capthick=2, color=colors[0])

    overlap = [correlation[1][0] for correlation in correlations_polar_10 if correlation[1][1] > threshold]
    for dot in overlap:
        plt.plot(1.1, dot, 'o', markersize=markersize, color=colors[1], alpha=alpha, label='10 µA')
    plt.plot(1.1, np.mean(overlap), 'o', markersize=markersize*2, color=colors[1])
    ax.errorbar(1.1, np.mean(overlap), yerr=np.std(overlap), capsize=20, capthick=2, color=colors[1])



    overlap = [correlation[1][0] for correlation in correlations_asymm_20 if correlation[1][1] > threshold]
    for dot in overlap:
        plt.plot(1.9, dot, 'o', markersize=markersize, color=colors[0], alpha=alpha)
    plt.plot(1.9, np.mean(overlap), 'o', markersize=markersize*2, color=colors[0])
    ax.errorbar(1.9, np.mean(overlap), yerr=np.std(overlap), capsize=20, capthick=2, color=colors[0])

    overlap = [correlation[1][0] for correlation in correlations_asymm_10 if correlation[1][1] > threshold]
    for dot in overlap:
        plt.plot(2.1, dot, 'o', markersize=markersize, color=colors[1], alpha=alpha)
    plt.plot(2.1, np.mean(overlap), 'o', markersize=markersize*2, color=colors[1])
    ax.errorbar(2.1, np.mean(overlap), yerr=np.std(overlap), capsize=20, capthick=2, color=colors[1])



    overlap = [correlation[1][0] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    for dot in overlap:
        plt.plot(2.9, dot, 'o', markersize=markersize, color=colors[0], alpha=alpha)
    plt.plot(2.9, np.mean(overlap), 'o', markersize=markersize*2, color=colors[0])
    ax.errorbar(2.9, np.mean(overlap), yerr=np.std(overlap), capsize=20, capthick=2, color=colors[0])

    overlap = [correlation[1][0] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    for dot in overlap:
        plt.plot(3.1, dot, 'o', markersize=markersize, color=colors[1], alpha=alpha)
    plt.plot(3.1, np.mean(overlap), 'o', markersize=markersize*2, color=colors[1])
    ax.errorbar(3.1, np.mean(overlap), yerr=np.std(overlap), capsize=20, capthick=2, color=colors[1])

    plt.axhspan(variability_overlaps_boundaries[0], variability_overlaps_boundaries[1], facecolor='grey', alpha=0.2)

    plt.ylim([0, 1])
    plt.yticks(fontsize=15)
    plt.ylabel('Binary overlap', fontsize=20)
    plt.xticks([1,2,3], ['Polarity', 'Symmetry', 'Both'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Neural overlap for different parameters', fontsize=20)
    fig.tight_layout()


    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (polarity 20 vs 10):")
    dataset1 = [correlation[1][0] for correlation in correlations_polar_20 if correlation[1][1] > threshold]
    dataset2 = [correlation[1][0] for correlation in correlations_polar_10 if correlation[1][1] > threshold]
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (asymmetry 20 vs 10):")
    dataset1 = [correlation[1][0] for correlation in correlations_asymm_20 if correlation[1][1] > threshold]
    dataset2 = [correlation[1][0] for correlation in correlations_asymm_10 if correlation[1][1] > threshold]
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (both 20 vs 10):")
    dataset1 = [correlation[1][0] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    dataset2 = [correlation[1][0] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (polarity 20uA vs biological variability):")
    dataset1 = [correlation[1][0] for correlation in correlations_polar_20 if correlation[1][1] > threshold]
    dataset2 = variability_overlaps
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (polarity 10uA vs biological variability):")
    dataset1 = [correlation[1][0] for correlation in correlations_polar_10 if correlation[1][1] > threshold]
    dataset2 = variability_overlaps
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (asymmetry 20uA vs biological variability):")
    dataset1 = [correlation[1][0] for correlation in correlations_asymm_20 if correlation[1][1] > threshold]
    dataset2 = variability_overlaps
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (asymmetry 10uA vs biological variability):")
    dataset1 = [correlation[1][0] for correlation in correlations_asymm_10 if correlation[1][1] > threshold]
    dataset2 = variability_overlaps
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (both 20uA vs biological variability):")
    dataset1 = [correlation[1][0] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    dataset2 = variability_overlaps
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for overlaps (both 10uA vs biological variability):")
    dataset1 = [correlation[1][0] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    dataset2 = variability_overlaps
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")



    fig, ax = plt.subplots()
    markersize=5
    colors = ['blue', 'orange']
    alpha = 0.5

    bias = [correlation[2] for correlation in correlations_polar_20 if correlation[1][1] > threshold] + [correlation[2] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    for centroid in bias:
        plt.plot(1, centroid, 'o', markersize=markersize, color=colors[0], alpha=alpha, label='20 µA')
    plt.plot(1, np.mean(bias), 'o', markersize=markersize*2, color=colors[0])
    ax.errorbar(1, np.mean(bias), yerr=np.std(bias), capsize=20, capthick=2, color=colors[0])

    bias = [correlation[2] for correlation in correlations_polar_10 if correlation[1][1] > threshold] + [correlation[2] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    for centroid in bias:
        plt.plot(2, centroid, 'o', markersize=markersize, color=colors[1], alpha=alpha, label='10 µA')
    plt.plot(2, np.mean(bias), 'o', markersize=markersize*2, color=colors[1])
    ax.errorbar(2, np.mean(bias), yerr=np.std(bias), capsize=20, capthick=2, color=colors[1])

    #plt.ylim([-0.1, 1.1])
    plt.yticks([-0.13, 0, 0.13, 0.27, 0.4], ['-20', '0', '20', '40', '60'], fontsize=15)
    plt.ylabel('Bias to cath.-first elec. [µm]', fontsize=20)
    plt.xticks([1,2], ['20 µA', '10 µA'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Centroid shift towards cathodic-first ', fontsize=20)
    plt.tight_layout()

    # Mann-Whitney U Rank-Sum test for unpaired data
    print("Rank Sum Mann-Whitney Test for polarity bias (20 vs 10):")
    dataset1 = [correlation[2] for correlation in correlations_polar_20 if correlation[1][1] > threshold] + [correlation[2] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    dataset2 = [correlation[2] for correlation in correlations_polar_10 if correlation[1][1] > threshold] + [correlation[2] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    u_statistic, u_pvalue = mannwhitneyu(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"U Statistic: {u_statistic}")
    print(f"P-Value: {u_pvalue}")

    # Wilcoxon Signed-Rank Test for data and a fixed value
    print("Wilcoxon Signed-Rank Test (fixed value) for polarity bias (20uA):")
    dataset1 = [correlation[2] for correlation in correlations_polar_20 if correlation[1][1] > threshold] + [correlation[2] for correlation in correlations_both_20 if correlation[1][1] > threshold]
    w_statistic, w_pvalue = wilcoxon([x - 0 for x in dataset1])
    ci = t.interval(0.95, len(dataset1)-1, loc=np.mean(dataset1), scale=sem(dataset1))
    print(f"Investigated dataset: {dataset1}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")
    print(f"95% confidence interval: {[x*150 for x in ci]}") # in um

    # Wilcoxon Signed-Rank Test for data and a fixed value
    print("Wilcoxon Signed-Rank Test (fixed value) for polarity bias (10uA):")
    dataset1 = [correlation[2] for correlation in correlations_polar_10 if correlation[1][1] > threshold] + [correlation[2] for correlation in correlations_both_10 if correlation[1][1] > threshold]
    w_statistic, w_pvalue = wilcoxon([x - 0 for x in dataset1])
    ci = t.interval(0.95, len(dataset1)-1, loc=np.mean(dataset1), scale=sem(dataset1))
    print(f"Investigated dataset: {dataset1}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")
    print(f"95% confidence interval: {[x*150 for x in ci]}") # in um


    ############################################################
    ######### DATA ANALYSIS DIRECTIONALITY CELLULAR ############
    ############################################################

    threshold_10 = 75
    threshold_20 = 100

    plt.figure()
    angles20 = []
    overlaps20 = []
    angles10 = []
    overlaps10 = []

    patterns = range(0, 13)

    for slice in list_directions20:
        angles, correlations, overlaps = slice.correlation_per_angle(patterns=patterns)
        for combination in range(len(angles)):
            if overlaps[combination][1] > threshold_20:
                angles20.append(angles[combination])
                overlaps20.append(overlaps[combination][0])
                plt.plot(angles[combination], overlaps[combination][0], 'o', markersize=markersize, color=colors[0], label='20 µA')

    for slice in list_directions10:
        angles, correlations, overlaps = slice.correlation_per_angle(patterns=patterns)
        for combination in range(len(angles)):
            if overlaps[combination][1] > threshold_10:
                angles10.append(angles[combination])
                overlaps10.append(overlaps[combination][0])
                plt.plot(angles[combination], overlaps[combination][0], 'o', markersize=markersize, color=colors[1], label='10 µA')

    # Compute the slope and intercept based on all the samples
    filtered_sorted_pairs20 = sorted((x,y) for x,y in zip(angles20, overlaps20) if not np.isnan(x))
    sorted_angles20, sorted_overlaps20 = zip(*filtered_sorted_pairs20)
    slope20, intercept20 = np.polyfit(sorted_angles20, sorted_overlaps20, 1)
    filtered_sorted_pairs10 = sorted((x,y) for x,y in zip(angles10, overlaps10) if not np.isnan(x))
    sorted_angles10, sorted_overlaps10 = zip(*filtered_sorted_pairs10)
    slope10, intercept10 = np.polyfit(sorted_angles10, sorted_overlaps10, 1)

    x_line = np.linspace(0, 180, 100)           # Generate 100 points between min and max of x
    y_line20 = slope20 * x_line + intercept20   # Compute corresponding y values
    y_line10 = slope10 * x_line + intercept10   # Compute corresponding y values
    linewidth=3
    plt.plot(x_line, y_line20, color='blue', linewidth=linewidth)
    equation_text = f"y = {slope20:.5f} x + {intercept20:.2f}"
    plt.text(0.05 * max(x_line), 0.9 * max(y_line20), equation_text, color='black', fontsize=10)
    plt.plot(x_line, y_line10, color='orange', linewidth=linewidth)
    equation_text = f"y = {slope10:.5f} x + {intercept10:.2f}"
    plt.text(0.05 * max(x_line), 0.9 * max(y_line10), equation_text, color='black', fontsize=10)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # To remove duplicate labels because of the for loop
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.xlabel('Angle between 2 directions [°]', fontsize=20)
    plt.ylabel('Binary overlap of 2 directions', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    bottom, top = plt.ylim()
    plt.ylim(bottom, 1)
    plt.title('Neural overlap for different directions', fontsize=20)
    plt.tight_layout()

    all_overlaps20 = []
    all_overlaps10 = []

    slice20_index = 0
    for slice in list_directions20:
        if slice20_index not in [0,1,3,5,9,13]:
            slice20_index += 1
            continue # skip this slice
        slice20_index += 1
        for i in range(slice.nb_patterns):
                for j in range(i+1,slice.nb_patterns):
                    all_overlaps20.append(slice.overlap(pattern1=patterns[i], pattern2=patterns[j]))

    for slice in list_directions10:
        for i in range(slice.nb_patterns):
                for j in range(i+1,slice.nb_patterns):
                    all_overlaps10.append(slice.overlap(pattern1=patterns[i], pattern2=patterns[j]))

    threshold_10 = 60
    threshold_20 = 80

    filtered_overlaps20 = []
    filtered_overlaps10 = []

    # Loop through both lists using the same index, with a being the overlap, and b being the number of active neurons
    for (a1, b1), (a2, b2) in zip(all_overlaps20, all_overlaps10):
        # Check conditions for filtering
        if b1 > threshold_20 and b2 > threshold_10:
            filtered_overlaps20.append(a1)  # Append a1 to filtered list
            filtered_overlaps10.append(a2)  # Append a2 to filtered list

    plt.figure()
    plt.axline(xy1=(0.4,0.4), slope=1, linewidth=2, color='blue')
    plt.scatter(filtered_overlaps20, filtered_overlaps10, c='blue', s=30, cmap='viridis')
    xlim = plt.xlim()
    ylim = plt.ylim()
    mintick = min([xlim[0], ylim[0]])
    maxtick = 1.01
    plt.xlim([mintick, maxtick])
    plt.ylim([mintick, maxtick])
    plt.text(mintick + 7/8*mintick/maxtick, mintick+0.02, 'r = ' + str(round(spearmanr(filtered_overlaps20, filtered_overlaps10, alternative='two-sided').statistic,2)), fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Neural overlap 20 µA', fontsize=20)
    plt.ylabel('Neural overlap 10 µA', fontsize=20)
    plt.gca().set_aspect('equal')
    plt.title('Neural overlap for different amplitudes', fontsize=20)
    plt.tight_layout()

    plt.figure()
    markersize=7
    for comparison in range(len(filtered_overlaps20)):
        plt.plot([1, 2], [filtered_overlaps20[comparison], filtered_overlaps10[comparison]], color='black', alpha=0.5)
        plt.plot(1, filtered_overlaps20[comparison], 'o', markersize=markersize, color='blue', label='20 µA')
        plt.plot(2, filtered_overlaps10[comparison], 'o', markersize=markersize, color='orange', label='10 µA')

    plt.yticks(fontsize=15)
    plt.ylabel('Neural overlap', fontsize=20)
    plt.ylim(top=1.01)
    plt.xticks([1,2], ['20 µA', '10 µA'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Neural overlap for different amplitudes', fontsize=20)
    plt.tight_layout()

    ############################################################
    ####### DATA ANALYSIS PRINCIPAL COMPONENT ANALYSIS #########
    ############################################################

    cumulative_variances10 = []
    for slice in list_directions10:
        cumulative_variance = slice.pca()
        if cumulative_variance is not None:
            cumulative_variances10.append(cumulative_variance)

    cumulative_variances20 = []
    for slice in list_directions20:
        cumulative_variance = slice.pca()
        if cumulative_variance is not None:
            cumulative_variances20.append(cumulative_variance)

    plt.figure()
    colors = ["red", "green", "blue", "lime", "cyan", "magenta", "orange", "purple", "brown", "pink", "yellow", "teal", "navy", "maroon"]
    color_index = 0
    for variance in cumulative_variances20:
        plt.plot(variance, linewidth=2.5, color=colors[color_index])
        color_index += 1

    plt.hlines(0.95, 0, 12, linestyles='--')
    plt.hlines(0.90, 0, 12, linestyles='--')

    plt.xlabel('Number of principal components', fontsize=20)
    plt.ylabel('Cumulative variance', fontsize=20)
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x+1), ',')))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('PCA on neural activity for 13 directions (20 µA)', fontsize=20)
    plt.tight_layout()

    plt.figure()
    colors = np.take(colors, [0,1,3,5,9,13])
    color_index = 0
    for variance in cumulative_variances10:
        plt.plot(variance, linewidth=2.5, color=colors[color_index])
        color_index += 1

    plt.hlines(0.95, 0, 12, linestyles='--')
    plt.hlines(0.90, 0, 12, linestyles='--')

    plt.xlabel('Number of principal components', fontsize=20)
    plt.ylabel('Cumulative variance', fontsize=20)
    plt.gca().get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x+1), ',')))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('PCA on neural activity for 13 directions (10 µA)', fontsize=20)
    plt.tight_layout()

    def thresholding_PCA(arrays, threshold):
        threshold_components = []
        for arr in arrays:
            idx = np.where(arr >= threshold)[0][0]
            y1 = arr[idx - 1]
            y0 = arr[idx]
            fraction = (threshold - y1)/(y0-y1)
            threshold_components.append(idx + fraction)
        return threshold_components, np.mean(threshold_components), np.std(threshold_components)

    print('Mean and std of PCA-components crossing the 90% and 95% threshold:')
    print(thresholding_PCA(cumulative_variances20, 0.90))
    print(thresholding_PCA(cumulative_variances20, 0.95))
    print(thresholding_PCA(cumulative_variances10, 0.90))
    print(thresholding_PCA(cumulative_variances10, 0.95))

    ############################################################
    ################ DATA ANALYSIS AMPLITUDE ###################
    ############################################################

    plt.figure()
    linewidth=2.5
    color='blue'
    alpha=0.5
    amp, neurons, _ = list_amplitude[0].amplitude_cumulative_activity(amp_range='low', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[2].amplitude_cumulative_activity(amp_range='all', numbering='old', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[3].amplitude_cumulative_activity(amp_range='all', numbering='old', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[4].amplitude_cumulative_activity(amp_range='high', numbering='old', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[5].amplitude_cumulative_activity(amp_range='high', numbering='old', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[6].amplitude_cumulative_activity(amp_range='all', numbering='old', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[7].amplitude_cumulative_activity(amp_range='all', numbering='old', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[8].amplitude_cumulative_activity(amp_range='all', numbering='new', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[9].amplitude_cumulative_activity(amp_range='all', numbering='new', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)
    amp, neurons, _ = list_amplitude[10].amplitude_cumulative_activity(amp_range='all', numbering='new', plot=False)
    plt.plot(amp, neurons, linewidth=linewidth, color=color, alpha=alpha)

    low_range_sum = [0, 0, 0, 0]
    high_range_sum = [0, 0, 0, 0]

    # Low range only
    for slice in [0]:
        amp, neurons, _ = list_amplitude[slice].amplitude_cumulative_activity(amp_range='low', numbering='old', plot=False)
        low_range_sum = [sum(x) for x in zip(low_range_sum, neurons)]

    # High range only
    for slice in [4, 5]:
        amp, neurons, _ = list_amplitude[slice].amplitude_cumulative_activity(amp_range='high', numbering='old', plot=False)
        high_range_sum = [sum(x) for x in zip(high_range_sum, neurons)]

    # Whole range, old numbering (with asymmetric pulses included)
    for slice in [2, 6, 7]:
        amp, neurons, _ = list_amplitude[slice].amplitude_cumulative_activity(amp_range='all', numbering='old', plot=False)
        low_range_sum = [sum(x) for x in zip(low_range_sum, neurons[0:4])]
        high_range_sum = [sum(x) for x in zip(high_range_sum, neurons[4:8])]

    # Whole range, new numbering (only symmetric pulses)
    for slice in [8, 9, 10]:
        amp, neurons, _ = list_amplitude[slice].amplitude_cumulative_activity(amp_range='all', numbering='new', plot=False)
        low_range_sum = [sum(x) for x in zip(low_range_sum, neurons[0:4])]
        high_range_sum = [sum(x) for x in zip(high_range_sum, neurons[4:8])]

    neurons = [x/7 for x in low_range_sum] + [x/8 for x in high_range_sum]

    plt.plot(amp, neurons, linewidth=5, color='blue')

    plt.xlabel('Current amplitude [µA]', fontsize=20)
    plt.ylabel('Nb. of active neurons', fontsize=20)
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40], fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Activity for increasing stim. amplitude', fontsize=20)
    plt.tight_layout()

    ############################################################
    ############ DATA ANALYSIS DEPTH AND INTRA-SHANK ###########
    ############################################################

    nbs = []
    centroids = []

    for slice in list_depth:
        for pattern in range(0,slice.nb_patterns):
            nb_active_neurons = slice.get_nb_active_neurons(pattern=pattern)[0]
            _, centroid, stdev, _, = slice.spatial_analysis(pattern=pattern, fitting_algorithm='gaussian', plot=False)
            nbs.append(nb_active_neurons)
            centroids.append(centroid)

    print('Overlap between 2 bipolar pairs with different electrodes (from depth experiment)')
    for slice in list_depth:
        print(slice.overlap(pattern1=1, pattern2=2, plot=False)) # Overlap between A in, B out & A out, B in (20 uA)
        print(slice.overlap(pattern1=0, pattern2=3, plot=False)) # Overlap between A in, B in & A out, B out (20 uA)
        print(slice.overlap(pattern1=5, pattern2=6, plot=False)) # Overlap between A in, B out & A out, B in (10 uA)
        print(slice.overlap(pattern1=4, pattern2=7, plot=False)) # Overlap between A in, B in & A out, B out (10 uA)

    plt.figure()
    markersize=7
    colors = ['blue', 'orange', 'blue', 'orange', 'blue', 'orange', 'blue', 'orange']

    for centroid in range(0, len(centroids), 4):
        plt.plot([1, 2, 3, 4], [centroids[centroid], centroids[centroid+1], centroids[centroid+2], centroids[centroid+3]], color='black', alpha=0.5)

    for centroid in range(0, len(centroids), 8):
        plt.plot(1, centroids[centroid], 'o', markersize=markersize, color=colors[0], label='20 µA')
    for centroid in range(4, len(centroids), 8):
        plt.plot(1, centroids[centroid], 'o', markersize=markersize, color=colors[1], label='10 µA')

    for centroid in range(1, len(centroids), 8):
        plt.plot(2, centroids[centroid], 'o', markersize=markersize, color=colors[2])
    for centroid in range(5, len(centroids), 8):
        plt.plot(2, centroids[centroid], 'o', markersize=markersize, color=colors[3])

    for centroid in range(2, len(centroids), 8):
        plt.plot(3, centroids[centroid], 'o', markersize=markersize, color=colors[4])
    for centroid in range(6, len(centroids), 8):
        plt.plot(3, centroids[centroid], 'o', markersize=markersize, color=colors[5])

    for centroid in range(3, len(centroids), 8):
        plt.plot(4, centroids[centroid], 'o', markersize=markersize, color=colors[6])
    for centroid in range(7, len(centroids), 8):
        plt.plot(4, centroids[centroid], 'o', markersize=markersize, color=colors[7])

    plt.ylim([-0.1, 1.1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0\nElec. A', '30', '60', '90', '120', 'Elec. B\n150'], fontsize=15)
    plt.ylabel('Location of centroid \nalong layer [µm]', fontsize=20)
    plt.xticks([1,2,3,4], ['A & B in', 'A in, B out', 'A out, B in', 'A & B out'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Centroid shift for depth stim.', fontsize=20)
    plt.tight_layout()

    plt.figure()
    markersize=7
    colors = ['blue', 'orange', 'blue', 'orange', 'blue', 'orange']

    for centroid in range(0, len(nbs), 4):
        plt.plot([1, 2, 3], [nbs[centroid], (nbs[centroid+1] + nbs[centroid+2])/2, nbs[centroid+3]], color='black', alpha=0.5)

    for centroid in range(0, len(nbs), 8):
        plt.plot(1, nbs[centroid], 'o', markersize=markersize, color=colors[0], label='20 µA')
    for centroid in range(4, len(nbs), 8):
        plt.plot(1, nbs[centroid], 'o', markersize=markersize, color=colors[1], label='10 µA')

    for centroid in range(1, len(nbs), 8):
        plt.plot(2, (nbs[centroid] + nbs[centroid+1])/2, 'o', markersize=markersize, color=colors[2])
    for centroid in range(5, len(nbs), 8):
        plt.plot(2, (nbs[centroid] + nbs[centroid+1])/2, 'o', markersize=markersize, color=colors[3])

    for centroid in range(3, len(nbs), 8):
        plt.plot(3, nbs[centroid], 'o', markersize=markersize, color=colors[4])
    for centroid in range(7, len(nbs), 8):
        plt.plot(3, nbs[centroid], 'o', markersize=markersize, color=colors[5])

    plt.yticks(fontsize=15)
    plt.ylabel('Nb. of activated neurons', fontsize=20)
    plt.xticks([1,2,3], ['A & B in-plane', '1 out', '2 out'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Amount of activity for depth stim.', fontsize=20)
    plt.tight_layout()


    nbs = []
    centroids = []
    stdevs = []

    for slice in list_local:
        for pattern in range(0,slice.nb_patterns):
            nb_active_neurons = slice.get_nb_active_neurons(pattern=pattern)[0]
            _, centroid, stdev, _, = slice.spatial_analysis(pattern=pattern, fitting_algorithm='gaussian', plot=False)
            nbs.append(nb_active_neurons)
            centroids.append(centroid)
            stdevs.append(stdev)

    plt.figure()
    markersize=7
    colors = ['blue', 'orange', 'blue', 'orange', 'blue', 'orange']

    for centroid in range(0, len(centroids), 3):
        plt.plot([1, 2, 3], [centroids[centroid], centroids[centroid+1], centroids[centroid+2]], color='black', alpha=0.5)

    for centroid in range(0, len(centroids), 6):
        plt.plot(1, centroids[centroid], 'o', markersize=markersize, color=colors[0], label='20 µA')
    for centroid in range(3, len(centroids), 6):
        plt.plot(1, centroids[centroid], 'o', markersize=markersize, color=colors[1], label='10 µA')

    for centroid in range(1, len(centroids), 6):
        plt.plot(2, centroids[centroid], 'o', markersize=markersize, color=colors[2])
    for centroid in range(4, len(centroids), 6):
        plt.plot(2, centroids[centroid], 'o', markersize=markersize, color=colors[3])

    for centroid in range(2, len(centroids), 6):
        plt.plot(3, centroids[centroid], 'o', markersize=markersize, color=colors[4])
    for centroid in range(5, len(centroids), 6):
        plt.plot(3, centroids[centroid], 'o', markersize=markersize, color=colors[5])

    plt.ylim([-0.1, 1.1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4], ['0\nElec. A', '30', '60', '90', '120', 'Elec. B\n150', '180', '210'], fontsize=15)
    plt.ylabel('Location of centroid \nalong layer [µm]', fontsize=20)
    plt.xticks([1,2,3], ['Regular (A to B)', 'Hyper-local (A)', 'Hyper-local (B)'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Centroid shift for hyper-local stim.', fontsize=20)
    plt.tight_layout()

    # Wilcoxon Signed-Rank Test for paired data
    print("Wilcoxon Signed-Rank Test (Paired Data) for hyper-local centroid shift (regular and A):")
    dataset1 = centroids[0:len(centroids):3]
    dataset2 = centroids[1:len(centroids):3]
    w_statistic, w_pvalue = wilcoxon(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")

    # Wilcoxon Signed-Rank Test for paired data
    print("Wilcoxon Signed-Rank Test (Paired Data) for hyper-local centroid shift (regular and B):")
    dataset1 = centroids[0:len(centroids):3]
    dataset2 = centroids[2:len(centroids):3]
    w_statistic, w_pvalue = wilcoxon(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")

    # Wilcoxon Signed-Rank Test for paired data
    print("Wilcoxon Signed-Rank Test (Paired Data) for hyper-local centroid shift (A and B):")
    dataset1 = centroids[1:len(centroids):3]
    dataset2 = centroids[2:len(centroids):3]
    w_statistic, w_pvalue = wilcoxon(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")

    # Wilcoxon Signed-Rank Test for data and a fixed value
    print("Wilcoxon Signed-Rank Test (fixed value) for hyper-local centroid shift (inter-shank):")
    dataset1 = centroids[0:len(centroids):3]
    w_statistic, w_pvalue = wilcoxon([x - 0.5 for x in dataset1])
    ci = t.interval(0.95, len(dataset1)-1, loc=np.mean(dataset1), scale=sem(dataset1))
    print(f"Investigated dataset: {dataset1}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")
    print(f"95% confidence interval: {[x*150 for x in ci]}") # in um

    # Wilcoxon Signed-Rank Test for data and a fixed value
    print("Wilcoxon Signed-Rank Test (fixed value) for hyper-local centroid shift (intra-shank A):")
    dataset1 = centroids[1:len(centroids):3]
    w_statistic, w_pvalue = wilcoxon([x - 0 for x in dataset1])
    ci = t.interval(0.95, len(dataset1)-1, loc=np.mean(dataset1), scale=sem(dataset1))
    print(f"Investigated dataset: {dataset1}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")
    print(f"95% confidence interval: {[x*150 for x in ci]}") # in um

    # Wilcoxon Signed-Rank Test for data and a fixed value
    print("Wilcoxon Signed-Rank Test (fixed value) for hyper-local centroid shift (intra-shank B):")
    dataset1 = centroids[2:len(centroids):3]
    w_statistic, w_pvalue = wilcoxon([x - 1 for x in dataset1])
    ci = t.interval(0.95, len(dataset1)-1, loc=np.mean(dataset1), scale=sem(dataset1))
    print(f"Investigated dataset: {dataset1}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")
    print(f"95% confidence interval: {[x*150 for x in ci]}") # in um


    plt.figure()
    markersize=7
    colors = ['blue', 'orange', 'blue', 'orange', 'blue', 'orange']

    for centroid in range(0, len(nbs), 3):
        plt.plot([1, 2], [nbs[centroid], (nbs[centroid+1] + nbs[centroid+2])/2], color='black', alpha=0.5)

    for centroid in range(0, len(nbs), 6):
        plt.plot(1, nbs[centroid], 'o', markersize=markersize, color=colors[0], label='20 µA')
    for centroid in range(3, len(nbs), 6):
        plt.plot(1, nbs[centroid], 'o', markersize=markersize, color=colors[1], label='10 µA')

    for centroid in range(1, len(nbs), 6):
        plt.plot(2, (nbs[centroid] + nbs[centroid+1])/2, 'o', markersize=markersize, color=colors[2])
    for centroid in range(4, len(nbs), 6):
        plt.plot(2, (nbs[centroid] + nbs[centroid+1])/2, 'o', markersize=markersize, color=colors[3])


    plt.yticks(fontsize=15)
    plt.ylabel('Nb. of activated neurons', fontsize=20)
    plt.xticks([1,2], ['Regular bipolar', 'Hyper-local'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # To remove duplicate labels because of the for loop
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Amount of activity for hyper-local stim.', fontsize=20)
    plt.tight_layout()

    # Wilcoxon Signed-Rank Test for paired data
    print("Wilcoxon Signed-Rank Test (Paired Data) for hyper-local activity amount (regular, hyper-local):")
    dataset1 = nbs[0:len(nbs):3]
    dataset2 = [sum(x)/2 for x in zip(nbs[1:len(nbs):3], nbs[2:len(nbs):3])]
    w_statistic, w_pvalue = wilcoxon(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")

    plt.figure()
    markersize=7
    colors = ['blue', 'orange', 'blue', 'orange', 'blue', 'orange']

    for centroid in range(0, len(nbs), 3):
        plt.plot([1, 2], [stdevs[centroid], (stdevs[centroid+1] + stdevs[centroid+2])/2], color='black', alpha=0.5)

    for centroid in range(0, len(nbs), 6):
        plt.plot(1, stdevs[centroid], 'o', markersize=markersize, color=colors[0], label='20 µA')
    for centroid in range(3, len(nbs), 6):
        plt.plot(1, stdevs[centroid], 'o', markersize=markersize, color=colors[1], label='10 µA')

    for centroid in range(1, len(nbs), 6):
        plt.plot(2, (stdevs[centroid] + stdevs[centroid+1])/2, 'o', markersize=markersize, color=colors[2])
    for centroid in range(4, len(nbs), 6):
        plt.plot(2, (stdevs[centroid] + stdevs[centroid+1])/2, 'o', markersize=markersize, color=colors[3])

    plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x*150), ','))) # scaling to get standard deviation in µm. 150 is distance between the two shanks in µm
    plt.yticks(fontsize=15)
    plt.ylabel('Std. dev. of neuronal spread [µm]', fontsize=20)
    plt.xticks([1,2], ['Regular bipolar', 'Hyper-local'], fontsize=15)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title('Extent of activity for hyper-local stim.', fontsize=20)
    plt.tight_layout()

    # Wilcoxon Signed-Rank Test for paired data
    print("Wilcoxon Signed-Rank Test (Paired Data) for hyper-local activity extent (regular, hyper-local):")
    dataset1 = stdevs[0:len(stdevs):3]
    dataset2 = [sum(x)/2 for x in zip(stdevs[1:len(stdevs):3], stdevs[2:len(stdevs):3])]
    w_statistic, w_pvalue = wilcoxon(dataset1, dataset2)
    print(f"Investigated datasets: {dataset1} and {dataset2}")
    print(f"N samples: {len(dataset1)}")
    print(f"W Statistic: {w_statistic}")
    print(f"P-Value: {w_pvalue}")

    plt.show()