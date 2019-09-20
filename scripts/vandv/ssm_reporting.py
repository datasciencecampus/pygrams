from statistics import stdev, mean

import numpy as np
import pandas as pd
from scipy.stats import trim_mean


def __html_table_from_dataframe(df, term_style='{:.0%}'):
    df_style = df.style.hide_index()
    df_style = df_style.set_table_styles([
        dict(selector='table', props=[('border-collapse', 'collapse')]),
        dict(selector='td', props=[('border', '2px solid black'),
                                   ('text-align', 'right'),
                                   ('padding-left', '15px'),
                                   ('padding-right', '15px')])
    ])
    for window_size in df.columns[1:]:
        df_style = df_style.format({window_size: term_style})
    df_style = df_style.highlight_max(axis=1)

    heading = '<style type="text/css">table {border-collapse: collapse;} </style>\n'
    return heading + df_style.render()


def __create_df_from_results(results):
    df_results = pd.DataFrame({'Terms': list(results.keys())})
    k_range = list(next(iter(results.values())).keys())

    for k in k_range:
        look_ahead_results = []
        for term_name in results:
            look_ahead_results.append(results[term_name][k]['accuracy'])

        df_term_column = pd.DataFrame({f'{k}': look_ahead_results})
        df_results = df_results.join(df_term_column)

    return df_results


def html_table(results):
    df_results = __create_df_from_results(results)
    results_table = __html_table_from_dataframe(df_results)
    return results_table


def trim_proportion(data, proportion_to_cut):
    # parts taken from scipy.stats.stats.trim_mean
    nobs = data.shape[0]
    lower_cut = int(proportion_to_cut * nobs)
    upper_cut = nobs - lower_cut
    if lower_cut > upper_cut:
        raise ValueError("Proportion too big.")

    data_tmp = np.partition(data, (lower_cut, upper_cut - 1), 0)

    sl = [slice(None)] * data_tmp.ndim
    sl[0] = slice(lower_cut, upper_cut)
    return data_tmp[tuple(sl)]


def summary_html_table(results, trimmed_proportion_to_cut=0.1):
    df_results = __create_df_from_results(results)

    means = {
        'Summary': f'<b>Mean</b>'}
    for prediction_length in df_results.columns[1:]:
        means[prediction_length] = mean(df_results[prediction_length])
    summary_df = pd.DataFrame(means, index=[0])

    trimmed_means = {
        'Summary': f'<b>Trimmed ({trimmed_proportion_to_cut * 100.0:.0f}% cut) mean</b>'}
    for prediction_length in df_results.columns[1:]:
        trimmed_means[prediction_length] = trim_mean(df_results[prediction_length],
                                                     trimmed_proportion_to_cut)
    summary_df = summary_df.append(trimmed_means, ignore_index=True)

    standard_deviations = {
        'Summary': f'<b>Standard deviation</b>'}
    for prediction_length in df_results.columns[1:]:
        standard_deviations[prediction_length] = stdev(df_results[prediction_length])
    summary_df = summary_df.append(standard_deviations, ignore_index=True)

    trimmed_standard_deviations = {
        'Summary': f'<b>Trimmed ({trimmed_proportion_to_cut * 100.0:.0f}% cut) standard deviation</b>'}
    for prediction_length in df_results.columns[1:]:
        trimmed_data = trim_proportion(df_results[prediction_length], trimmed_proportion_to_cut)
        trimmed_standard_deviations[prediction_length] = stdev(trimmed_data)
    summary_df = summary_df.append(trimmed_standard_deviations, ignore_index=True)

    summary_table = __html_table_from_dataframe(summary_df)

    return summary_table
