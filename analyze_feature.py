# Code by Jonatan Bonjour

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
from scipy.stats import mannwhitneyu
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, proportional_hazard_test
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from fpdf import FPDF


def hist_plot(dataframe, feature, save_dir_path=None, bins=10):
    """Plot a histogram for the given feature"""
    plt.figure(figsize=(8, 5))
    sns.histplot(dataframe, x=feature, bins=bins)
    plt.title(f'Histogram')
    plt.tight_layout()
    if save_dir_path is not None:
        plt.savefig(f'{save_dir_path}/hist_{feature}.png', dpi=200)
    plt.show()


def violin_plot(dataframe, feature, save_dir_path=None, smoothing=0.3):
    """Plot a violin plot for the given feature"""
    palette = {0: '#ff7f0e', 1: '#1f77b4'}
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='response', y=feature, data=dataframe, palette=palette, bw_adjust=smoothing, inner='quartile')
    plt.title(f'Violin plots')
    plt.tight_layout()
    if save_dir_path is not None:
        plt.savefig(f'{save_dir_path}/violin_{feature}.png', dpi=200)
    plt.show()


def waterfall_plot(dataframe, feature, save_dir_path=None):
    """Waterfall plot for the given feature"""
    df = dataframe.dropna(subset=['response']).reset_index(drop=True)
    sorted_df = df.sort_values(by=feature, ascending=True)
    palette = {1: '#1f77b4', 0: '#ff7f0e'}
    plt.figure(figsize=(8, 5))

    plt.bar(sorted_df['patient_id'], sorted_df[feature], color=sorted_df['response'].map(palette), width=1)

    plt.xlabel('Patients')
    plt.ylabel(feature)
    plt.xticks(rotation=90)

    true_patch = mpatches.Patch(color=palette[True], label='Responders')
    false_patch = mpatches.Patch(color=palette[False], label='Non-responders')
    plt.legend(handles=[true_patch, false_patch])

    plt.title(f'Waterfall plot')
    plt.tight_layout()
    if save_dir_path is not None:
        plt.savefig(f'{save_dir_path}/waterfall_{feature}.png', dpi=200)

    plt.show()


def km_plot(dataframe, feature, save_dir_path=None, num_curves=2, type='os', show_ci=False, ticks=12):
    """Plot Kaplan-Meier curves for the given feature, split into specified number of groups."""
    df = dataframe.dropna(subset=[f'{type}_event', f'{type}_months']).reset_index(drop=True)

    # Calculate quantile thresholds for splitting data
    thresholds = np.quantile(df[feature], np.linspace(0, 1, num_curves + 1))

    plt.figure(figsize=(8, 5))
    curves = []

    for i in range(num_curves):
        # Select data for each group
        if i == 0:
            group = df[df[feature] < thresholds[i + 1]]
        elif i == num_curves - 1:
            group = df[df[feature] >= thresholds[i]]
        else:
            group = df[(df[feature] >= thresholds[i]) & (df[feature] < thresholds[i + 1])]

        # Fit and plot KM curve for the group
        kmf = KaplanMeierFitter()
        kmf.fit(group[f'{type}_months'], event_observed=group[f'{type}_event'], label=f'Group {i + 1}')
        kmf.plot(show_censors=True, ci_show=show_ci)
        curves.append(kmf)

    plt.xlabel('Months')
    plt.ylabel('Probability of survival')
    plt.ylim(0, 1.05)
    plt.gca().xaxis.set_major_locator(MultipleLocator(ticks))
    add_at_risk_counts(*curves, rows_to_show=['At risk'])

    plt.title(f'Kaplan-Meier curves ({num_curves}) {type.upper()}')
    plt.tight_layout()
    fig = plt.gcf()
    if save_dir_path is not None:
        fig.savefig(f'{save_dir_path}/km_{type}_{num_curves}_curves_{feature}.png', dpi=200)
    plt.show()


def mann_whitney_p(dataframe, feature):
    """Compute the Mann-Whitney U test for the given feature (response 0 vs 1) and returns the p-value"""
    df = dataframe.dropna(subset=['response']).reset_index(drop=True)
    table1 = df.loc[(df['response'] == 1)][feature]
    table2 = df.loc[(df['response'] == 0)][feature]
    stat, p = mannwhitneyu(table1, table2)
    return p


def logranktest(dataframe, feature, type='os'):
    """Compute the logrank test for the given feature and returns the p-value"""
    df = dataframe.dropna(subset=[f'{type}_event', f'{type}_months']).reset_index(drop=True)
    median_value = dataframe[feature].median()
    table1 = df.loc[(df[feature] >= median_value)]
    table2 = df.loc[(df[feature] < median_value)]

    lrt = logrank_test(
        durations_A=table1[f'{type}_months'],
        durations_B=table2[f'{type}_months'],
        event_observed_A=table1[f'{type}_event'],
        event_observed_B=table2[f'{type}_event']
    )
    return lrt.p_value


def cph_univariable_analysis(dataframe, feature, type='os', use_dichotomization=False, scaling_method='standardize'):
    """For the Cox proportional hazard univariable analysis on a single feature,
    with an option to use dichotomization of the feature around the median value,
    and an option to standardize or normalize the feature.
    It also includes testing the proportional hazards assumption using Schoenfeld residuals."""

    df = dataframe.dropna(subset=[f'{type}_event', f'{type}_months']).reset_index(drop=True)

    # Apply scaling if requested
    if scaling_method:
        if scaling_method == 'standardize':
            scaler = StandardScaler()
        elif scaling_method == 'normalize':
            scaler = MinMaxScaler()
        else:
            raise ValueError('Invalid scaling method')
        df[feature] = scaler.fit_transform(df[[feature]])

    # Calculate median and dichotomize the feature if required
    if use_dichotomization:
        median_value = df[feature].median()
        df[f'{feature}_binary'] = df[feature].apply(lambda x: 1 if x >= median_value else 0)
        feature_to_use = f'{feature}_binary'
    else:
        feature_to_use = feature

    # Fit Cox proportional hazard model with the chosen feature
    results_dict = {}
    cphf = CoxPHFitter(penalizer=0.0001)
    df_filtered = df[[f'{type}_months', f'{type}_event', feature_to_use]].dropna()
    cphf.fit(df_filtered, duration_col=f'{type}_months', event_col=f'{type}_event')
    result = cphf.summary[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p', 'coef', 'coef lower 95%', 'coef upper 95%']]

    # Store results in a dictionary
    for col in result.columns:
        results_dict[col] = result[col].values[0]
    results_dict['coef error'] = (result['coef upper 95%'].values[0] - result['coef lower 95%'].values[0]) / 2

    # Test proportional hazards assumption for the chosen feature and store p-value
    test_result = proportional_hazard_test(cphf, df_filtered, time_transform='rank')
    results_dict['assumption p'] = test_result.p_value[0]

    return results_dict


def forest_plot(dataframe, feature, save_dir_path=None, scale='linear', scaling_method='standardize'):
    """Forest plot for the Cox proportional hazard univariable analysis on a single feature."""

    # Perform the analysis for each configuration
    results_os = cph_univariable_analysis(dataframe, feature, type='os', scaling_method=scaling_method,
                                          use_dichotomization=False)
    results_os_dich = cph_univariable_analysis(dataframe, feature, type='os', use_dichotomization=True)
    results_pfs = cph_univariable_analysis(dataframe, feature, type='pfs', scaling_method=scaling_method,
                                           use_dichotomization=False)
    results_pfs_dich = cph_univariable_analysis(dataframe, feature, type='pfs', use_dichotomization=True)

    # Prepare data for the forest plot
    scaling_name = f"{scaling_method}d" if scaling_method is not None else 'no scaling'
    forest_data = {
        'Analysis': [f'OS {scaling_name}', 'OS dichotomized', f'PFS {scaling_name}', 'PFS dichotomized'],
        'HR': [results_os['exp(coef)'], results_os_dich['exp(coef)'],
               results_pfs['exp(coef)'], results_pfs_dich['exp(coef)']],
        'lower_CI': [results_os['exp(coef) lower 95%'], results_os_dich['exp(coef) lower 95%'],
                     results_pfs['exp(coef) lower 95%'], results_pfs_dich['exp(coef) lower 95%']],
        'upper_CI': [results_os['exp(coef) upper 95%'], results_os_dich['exp(coef) upper 95%'],
                     results_pfs['exp(coef) upper 95%'], results_pfs_dich['exp(coef) upper 95%']]
    }
    forest_df = pd.DataFrame(forest_data)

    plt.figure(figsize=(8, 5))

    plt.errorbar(forest_df['HR'], forest_df['Analysis'], xerr=(forest_df['HR'] - forest_df['lower_CI'],
                                                               forest_df['upper_CI'] - forest_df['HR']), fmt='o')
    plt.xscale(scale)
    plt.xlabel('HR')
    plt.title('Forest plot of univariable Cox analysis')
    plt.axvline(x=1, color='grey', linestyle='--')
    plt.tight_layout()

    if save_dir_path is not None:
        plt.savefig(f'{save_dir_path}/forest_{feature}.png', dpi=200)
    plt.show()


def logistic_regression_loocv_auc(dataframe, feature, scaling_method='standardize'):
    """AUC from logistic regression with leave-one-out cross-validation"""
    df = dataframe.dropna(subset=['response']).reset_index(drop=True)
    X = df[[feature]]
    y = df['response'].astype(int)

    probabilities = []
    loo = LeaveOneOut()

    # Loop over each split
    for train_index, test_index in loo.split(X):
        # Split data into training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = (y.iloc[train_index], y.iloc[test_index])

        # Scale data
        if scaling_method:
            if scaling_method == 'standardize':
                scaler = StandardScaler()
            elif scaling_method == 'normalize':
                scaler = MinMaxScaler()
            else:
                raise ValueError('Invalid scaling method')
            X_train.loc[:, feature] = scaler.fit_transform(X_train[[feature]]).ravel()
            X_test.loc[:, feature] = scaler.transform(X_test[[feature]]).ravel()

        # Initialize and fit logistic regression model
        model = LogisticRegression(penalty='l2', class_weight='balanced',
                                   fit_intercept=True, solver='liblinear', max_iter=100)
        model.fit(X_train, y_train)

        # Predict probability
        probability = model.predict_proba(X_test)
        probabilities.append(probability[0][1])

    # Calculate AUC
    auc = roc_auc_score(y, probabilities)

    return auc


def compute_stats(dataframe, feature, scaling_method='standardize'):
    """Compute statistics for a feature"""
    results = {}
    results['Mann-Whitney p-value'] = mann_whitney_p(dataframe, feature)
    results['Logrank OS p-value'] = logranktest(dataframe, feature)
    results['Logrank PFS p-value'] = logranktest(dataframe, feature, type='pfs')
    cox_os = cph_univariable_analysis(dataframe, feature, type='os', scaling_method=scaling_method)
    cox_pfs = cph_univariable_analysis(dataframe, feature, type='pfs', scaling_method=scaling_method)
    cox_os_dichotomized = cph_univariable_analysis(dataframe, feature, use_dichotomization=True)
    cox_pfs_dichotomized = cph_univariable_analysis(dataframe, feature, type='pfs', use_dichotomization=True)
    results['Cox OS p-value'] = cox_os['p']
    results['Cox OS HR'] = cox_os['exp(coef)']
    results['Cox OS assumption p-value'] = cox_os['assumption p']
    results['Cox PFS p-value'] = cox_pfs['p']
    results['Cox PFS HR'] = cox_pfs['exp(coef)']
    results['Cox PFS assumption p-value'] = cox_pfs['assumption p']
    results['Cox OS dichotomized p-value'] = cox_os_dichotomized['p']
    results['Cox OS dichotomized HR'] = cox_os_dichotomized['exp(coef)']
    results['Cox OS dichotomized assumption p-value'] = cox_os_dichotomized['assumption p']
    results['Cox PFS dichotomized p-value'] = cox_pfs_dichotomized['p']
    results['Cox PFS dichotomized HR'] = cox_pfs_dichotomized['exp(coef)']
    results['Cox PFS dichotomized assumption p-value'] = cox_pfs_dichotomized['assumption p']
    results['Logistic regression LOOCV AUC'] = logistic_regression_loocv_auc(dataframe, feature,
                                                                             scaling_method=scaling_method)
    return results


def analyze_feature(dataframe, feature, save_dir_path, scaling_method='standardize'):
    """
    Analyze a specific feature in a given dataframe and save the analysis results.

    This function performs various statistical analyses and visualizations on a specified feature in the dataframe. It generates histogram, violin, waterfall, and Kaplan-Meier plots, computes statistics, and then consolidates all the results into a PDF file saved in the specified directory.

    Parameters:
    - dataframe (pd.DataFrame): The dataframe containing the data to be analyzed. Must include the columns 'patient_id', 'response', 'os_months', 'os_event', 'pfs_months', and 'pfs_event', in addition to the features.
    - feature (str): The name of the feature (column in the dataframe) to be analyzed.
    - save_dir_path (str): The file path where the resulting plots and PDF file will be saved.
    """

    # Create plots and compute statistics
    hist_plot(dataframe, feature, save_dir_path)
    violin_plot(dataframe, feature, save_dir_path)
    waterfall_plot(dataframe, feature, save_dir_path)
    km_plot(dataframe, feature, save_dir_path, 2, 'os')
    km_plot(dataframe, feature, save_dir_path, 3, 'os')
    km_plot(dataframe, feature, save_dir_path, 2, 'pfs')
    km_plot(dataframe, feature, save_dir_path, 3, 'pfs')
    forest_plot(dataframe, feature, save_dir_path, scaling_method=scaling_method)
    stats = compute_stats(dataframe, feature, scaling_method=scaling_method)
    print(stats)

    # Create PDF file
    pdf = FPDF()
    pdf.set_top_margin(3)
    pdf.set_auto_page_break(auto=1, margin=1)
    pdf.add_page()
    pdf.set_font('Arial', '', 16)
    pdf.cell(0, 10, f'Features analysis: {feature}', align='C')
    pdf.ln()
    current_y = pdf.get_y()
    pdf.image(f'{save_dir_path}/hist_{feature}.png', y=current_y, w=90, h=0)
    pdf.image(f'{save_dir_path}/violin_{feature}.png', x=110, y=current_y, w=90, h=0)
    pdf.ln(50 + 5)
    current_y = pdf.get_y()
    pdf.image(f'{save_dir_path}/waterfall_{feature}.png', y=current_y, w=90, h=0)
    pdf.image(f'{save_dir_path}/forest_{feature}.png', x=110, y=current_y, w=90, h=0)
    pdf.ln(50 + 5)
    current_y = pdf.get_y()
    pdf.image(f'{save_dir_path}/km_os_2_curves_{feature}.png', y=current_y, w=90, h=0)
    pdf.image(f'{save_dir_path}/km_pfs_2_curves_{feature}.png', x=110, y=current_y, w=90, h=0)
    pdf.ln(50 + 5)
    current_y = pdf.get_y()
    pdf.image(f'{save_dir_path}/km_os_3_curves_{feature}.png', y=current_y, w=90, h=0)
    pdf.image(f'{save_dir_path}/km_pfs_3_curves_{feature}.png', x=110, y=current_y, w=90, h=0)
    pdf.ln(50 + 5)
    pdf.set_font('Arial', '', 10)
    scaling_name = f"{scaling_method.capitalize()}d" if scaling_method is not None else 'Not scaled'
    pdf.cell(40, 10, f'{scaling_name}:', ln=0)
    pdf.cell(70, 10, f'Cox OS HR: {round(stats["Cox OS HR"], 5)}', ln=0)
    pdf.cell(70, 10, f'Cox PFS HR: {round(stats["Cox PFS HR"], 5)}')
    pdf.ln(5)
    pdf.cell(40, 10, f'', ln=0)
    pdf.cell(70, 10, f'Cox OS p-value: {round(stats["Cox OS p-value"], 5)}', ln=0)
    pdf.cell(70, 10, f'Cox PFS p-value: {round(stats["Cox PFS p-value"], 5)}')
    pdf.ln(5)
    pdf.cell(40, 10, f'', ln=0)
    pdf.cell(70, 10, f'Cox OS ph assumption p-value: {round(stats["Cox OS assumption p-value"], 5)}', ln=0)
    pdf.cell(70, 10, f'Cox PFS ph assumption p-value: {round(stats["Cox PFS assumption p-value"], 5)}')
    pdf.ln(7)
    pdf.cell(40, 10, f'Dichotomized:', ln=0)
    pdf.cell(70, 10, f'logrank OS p-value: {round(stats["Logrank OS p-value"], 5)}', ln=0)
    pdf.cell(70, 10, f'logrank PFS p-value: {round(stats["Logrank PFS p-value"], 5)}')
    pdf.ln(5)
    pdf.cell(40, 10, f'', ln=0)
    pdf.cell(70, 10, f'Cox OS HR: {round(stats["Cox OS dichotomized HR"], 5)}', ln=0)
    pdf.cell(70, 10, f'Cox PFS HR: {round(stats["Cox PFS dichotomized HR"], 5)}')
    pdf.ln(5)
    pdf.cell(40, 10, f'', ln=0)
    pdf.cell(70, 10, f'Cox OS p-value: {round(stats["Cox OS dichotomized p-value"], 5)}', ln=0)
    pdf.cell(70, 10, f'Cox PFS p-value: {round(stats["Cox PFS dichotomized p-value"], 5)}')
    pdf.ln(5)
    pdf.cell(40, 10, f'', ln=0)
    pdf.cell(70, 10, f'Cox OS ph assumption p-value: {round(stats["Cox OS dichotomized assumption p-value"], 5)}', ln=0)
    pdf.cell(70, 10, f'Cox PFS ph assumption p-value: {round(stats["Cox PFS dichotomized assumption p-value"], 5)}')
    pdf.ln(7)
    pdf.cell(40, 10, f'Distribution:', ln=0)
    pdf.cell(70, 10, f'Mann-Whitney p-value: {round(stats["Mann-Whitney p-value"], 5)}')
    pdf.ln(7)
    pdf.cell(40, 10, f'Prediction:', ln=0)
    pdf.cell(70, 10, f'AUC logistic regression LOOCV: {round(stats["Logistic regression LOOCV AUC"], 5)}')
    pdf.ln(5)
    pdf.output(f'{save_dir_path}/{feature}.pdf', 'F')
    print(f'\n PDF saved in {save_dir_path}/{feature}.pdf')