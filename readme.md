# predicting auxilin spikes in clathrin-mediated endocytosis

![](https://yu-group.github.io/auxilin-prediction/reports/figs/fig_pipeline.jpg)

## quickstart
- **download data**: download cached data after tracking from [this gdrive folder](https://drive.google.com/drive/folders/1mh2wn0KLtj90j_rfgPUGEJgMZAEl0Yi7?usp=sharing) - should be added to the folder `data/tracks`
- **process data**: run `python data.py` to properly preprocess all the data (will cache it into the "processed folder")
- **rerun analysis**: [notebooks](notebooks) folder contains step-by-step analysis
- **tests**: run tests with `pytest` in the `tests` folder

## acknowledgements
- this is a collaboration between the Berkeley [Yu-Group](https://www.stat.berkeley.edu/~yugroup/) and the [Berkeley Advanced Bioimaging Center](http://abc.berkeley.edu/)
- uses code from several wonderful packages including [cmeAnalysis](https://github.com/DanuserLab/cmeAnalysis) (DanuserLab)