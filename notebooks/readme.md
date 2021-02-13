This is an overview of the markdown contents of all the notebooks / scripts in this directory.

# 00_eda_images


#### TP/FP videos

## kymographs

# 00_eda_tracks


## look at tracks themselves

## visualize a single example

## basic eda

## look at displacement features

## look at learned dictionaries

## viz tracks where auxilin peaks first

## visualize splines

## feats centered around peak
**gets the clathrin traces aligned by their maximum, padded by zeros when necessary**

## pairplot of most relevant feats

## spike-times plot

## spike magnitude plot

###### Random projections

# 01_reg


# 01_reg_amplitude


#### compare with DASC

###### Max strength prediction

# 01_reg_timing


# 02_defn_y_manual


## unsure tracks

## visualize outcomes
- take mean, take sigma, define events with a threshold

## viz curves

**compare as a func of aux_max**

## look at hotspots

## compare different outcome defs

## analyze aux+ p-value / amplitude defs

## label checking

# 02_defn_y_rules


## unsure tracks

## rest of nb is finding rules to define outcome

# 02_run_data_pipeline


## process original data

## process new data

## check data stats

## visualize differences

# 03_classify


# 04_analyze_results


**baseline stats**

**look at prediction metrics**

**look at feat importances**

# 05_calc_dists


# 06_eda_limits


**binarize features**

**acc predicting majority class for every bucket**

# 07_eda_dynamin


## load dynamin tracks directly

**look at lifetimes**

**look at clath curves**

**compare aux + dynamin**

# 08_fit_dnn_tracks


## prepare data

#### train neural net with cv

**analyze cv results**

#### train neural net on all training data

## transfer to classification

## plot tracks where lstm does better than gb

# 09_fit_dnn_video


## prepare data

## show an example

## fit the data

## transfer to classification

# 10_interpret_scalars


## prepare data

## prepare for interp

## important features

## global feat imp

# 11_interpret_interactions


## prepare data

## segmentation

**make simple illustrative plot**

**recalculate with normalized scores (there is some err here)**

## interpret one pred at multiple scales

# 12_interpret_pred_plots


## prepare data

## prediction plots

## scatter plots

## viz errs

## viz total acc

# generate_full_data_results


