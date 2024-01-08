# Feature Analysis

## Dependencies
You need the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- lifelines
- sklearn
- fpdf

Make sure these are installed in your Python environment.

## Overview of functionalities

1. **Plotting functions**
   - `hist_plot()`: Generates histograms.
   - `violin_plot()`: Creates violin plots.
   - `waterfalls_plot()`: Constructs waterfall plots.
   - `km_plot()`: Plots Kaplan-Meier survival curves.

2. **Statistical analysis functions**
   - `mann_whitney_p()`: Computes the Mann-Whitney U test p-value.
   - `logranktest()`: Performs the log-rank test for survival data.
   - `cph_univariable_analysis()`: Conducts univariable Cox proportional hazards analysis.
   - `logistic_regression_loocv_auc()`: Calculates the AUC from logistic regression with leave-one-out cross-validation.

3. **Comprehensive analysis function**
   - `analyze_feature()`: This function conducts a thorough analysis of a specific feature, including generating plots, computing statistics, and compiling the results into a PDF report.

## Usage
Refer to the example provided in the example_notebook. Use the analyze_feature() function, passing in your DataFrame, the specific feature you wish to analyze, and the directory path where you want the output (plots and pdf) to be saved.



