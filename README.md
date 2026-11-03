# Bipolar Intracortical Microstimulation: analysis code
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23889569.svg)](https://doi.org/10.5281/zenodo.23889569)

Analysis code for the two-photon calcium imaging experiments in:

> Schelles M, Verhaege A, Van Rompaey N, Goyvaerts L, Wierda K, Ceyssens F, Kraft M, Bonin V. Bipolar intracortical microstimulation expands the repertoire of neuronal activation patterns. *Journal of Neural Engineering* (2026). https://doi.org/10.1088/1741-2552/aebb67

Maarten Schelles, ReVision Implant, maarten@revision-implant.com

## Contents

- `neuron_activity_processing.py`
  - `ExtractFluorescence`: ΔF/F trace per cell (mean over the pixels of its mask), peak response per stimulation pattern, activity threshold and cell centroid. Results are cached as `cell_data_<name>` in the data directory.
  - `ProcessFluorescence`: comparison of stimulation patterns (binary overlap, Spearman correlation, angle between current directions, PCA) and the center and extent of activity along an axis (Gaussian fit, weighted mean or kernel density estimate).
  - `get_all_electrode_locations`: positions of all return electrodes, derived from three annotated electrode positions.
  - Analysis section, executed when the file is run as a script, that produces the summary figures and statistical tests of the paper.
- `requirements.txt`: package versions used for the analysis.

## Installation

Python 3.9 (the analysis was run with Python 3.9.19):

```
pip install -r requirements.txt
```

## Input data

The imaging data are not included in this repository; see the data availability statement of the paper. For each dataset, `ExtractFluorescence` expects two files in the same directory:

- `<name>_stimall_seg.npy`: Cellpose segmentation (dictionary with key `masks`)
- `<name>_dff.tif`: ΔF/F stack (time × height × width, see Methods of the paper), 72 frames (8 s at 9 Hz) per stimulation pattern, patterns concatenated in time

## Usage

```python
from neuron_activity_processing import ExtractFluorescence, ProcessFluorescence, get_all_electrode_locations

experiment = ExtractFluorescence(directory="path/to/data", recording_name="name", nb_patterns=13)
central, column, layer = (200, 150), (100, 250), (300, 250)  # electrode positions in pixels
electrode_locations = [central, get_all_electrode_locations(central, column, layer), column, layer]
dataset = ProcessFluorescence(experiment.cell_data, electrode_locations)
print(dataset.overlap(pattern1=0, pattern2=1))  # (binary overlap, number of cells active for either pattern)
```

To rerun the analysis of the paper, add the datasets to the lists in the DATA LOADING section of `neuron_activity_processing.py` and run:

```
python neuron_activity_processing.py
```

## Citation

If you use this code, please cite the paper above. The version of the code used in the paper is archived on Zenodo: https://doi.org/10.5281/zenodo.23889569

## License

MIT, see [LICENSE](LICENSE).
