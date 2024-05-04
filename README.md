# data-lifecycle

Incldues the following folders:

1. `data_extraction` - Build datasets from the basis ones
2. `data_analysis` - Perform data analysis on the plots

# Usage

1. Notebooks for a given dataset need to have the same name as the dataset in all folders mentioned above

2. Use the `data_analysis/plotter` utility to display and save the plots in the appropriate structure to make use of the automatic upload pipeline.

## Uploading a dataset to HuggingFace

Once the dataset is built use `upload.ipynb` to upload the dataset.