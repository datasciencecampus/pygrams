from statistics import stdev, mean

import numpy as np
import pandas as pd
from scipy.stats import trim_mean


def __html_table_from_dataframe(df_results, terms, term_style='{:.0f}'):
    df_table = df_results.style.hide_index()
    df_table = df_table.set_table_styles([
        dict(selector='table', props=[('border-collapse', 'collapse')]),
        dict(selector='td', props=[('border', '2px solid black'),
                                   ('text-align', 'right'),
                                   ('padding-left', '15px'),
                                   ('padding-right', '15px')])
    ])
    for term in terms:
        df_table = df_table.format({term: term_style})
    df_table = df_table.highlight_max(axis=1)

    heading = '<style type="text/css">table {border-collapse: collapse;} </style>\n'
    return heading + df_table.render()


def __create_df_from_results(prediction_lengths, results):
    df_results = pd.DataFrame({'terms': list(results.keys())})
    for prediction_length in prediction_lengths:
        prediction_length_results = []
        for term_name in results:
            prediction_length_results.append(results[term_name][prediction_length])

        df_term_column = pd.DataFrame({f'{prediction_length}': prediction_length_results})
        df_results = df_results.join(df_term_column)
    return df_results


def html_table(results, prediction_lengths):
    df_results = __create_df_from_results(prediction_lengths, results)

    results_table = __html_table_from_dataframe(df_results, 'SSM')

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


def summary_html_table(results, prediction_lengths, trimmed_proportion_to_cut=0.1):
    df_results = __create_df_from_results(prediction_lengths, results)

    means = {
        'terms': f'<b>Mean</b>'}
    for prediction_length in prediction_lengths:
        means[f'{prediction_length}'] = mean(df_results[f'{prediction_length}'])
    summary_df = pd.DataFrame(means, index=[0])

    trimmed_means = {
        'terms': f'<b>Trimmed ({trimmed_proportion_to_cut * 100.0:.0f}% cut) mean</b>'}
    for prediction_length in prediction_lengths:
        trimmed_means[f'{prediction_length}'] = trim_mean(df_results[f'{prediction_length}'],
                                                          trimmed_proportion_to_cut)
    summary_df = summary_df.append(trimmed_means, ignore_index=True)

    standard_deviations = {
        'terms': f'<b>Standard deviation</b>'}
    for prediction_length in prediction_lengths:
        standard_deviations[f'{prediction_length}'] = stdev(df_results[f'{prediction_length}'])
    summary_df = summary_df.append(standard_deviations, ignore_index=True)

    trimmed_standard_deviations = {
        'terms': f'<b>Trimmed ({trimmed_proportion_to_cut * 100.0:.0f}% cut) standard deviation</b>'}
    for prediction_length in prediction_lengths:
        trimmed_data = trim_proportion(df_results[f'{prediction_length}'], trimmed_proportion_to_cut)
        trimmed_standard_deviations[f'{prediction_length}'] = stdev(trimmed_data)
    summary_df = summary_df.append(trimmed_standard_deviations, ignore_index=True)

    summary_table = __html_table_from_dataframe(summary_df, 'SSM summary')

    return summary_table
