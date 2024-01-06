# Feature Analysis Notebook

## Dependencies
To run this notebook, you need the following Python libraries:
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
The notebook includes several key functions, each designed to perform specific types of analysis:

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
- Import the required libraries and set up the notebook environment.
- Load or create your DataFrame. If you are using your own data, ensure it has the necessary columns as shown in the sample DataFrame.
- Call the `analyze_feature()` function with your DataFrame, the feature you want to analyze, and the directory path to save the output.

## Output
- The notebook saves various plots and a PDF report containing the analysis results in the specified directory.

