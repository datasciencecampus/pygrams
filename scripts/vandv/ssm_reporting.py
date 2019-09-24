from statistics import stdev, mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from tqdm import tqdm

from scripts.vandv.graphs import plot_to_html_image


def __html_table_from_dataframe(df, term_style='{:.0%}', highlight_max=True, first_col=1):
    df_style = df.style.hide_index()
    df_style = df_style.set_table_styles([
        dict(selector='table', props=[('border-collapse', 'collapse')]),
        dict(selector='td', props=[('border', '2px solid black'),
                                   ('text-align', 'right'),
                                   ('padding-left', '15px'),
                                   ('padding-right', '15px')])
    ])

    for window_size in df.columns[first_col:]:
        df_style = df_style.format({window_size: term_style})

    if highlight_max:
        df_style = df_style.highlight_max(axis=1)
    else:
        df_style = df_style.highlight_min(axis=1)

    heading = '<style type="text/css">table {border-collapse: collapse;} </style>\n'
    return heading + df_style.render()


def __create_df_from_results(results):
    df_results = pd.DataFrame(results).T
    df_results.reset_index(level=0, inplace=True)
    # k_range = list(next(iter(results.values())).keys())
    #
    # for k in k_range:
    #     look_ahead_results = []
    #     for term_name in results:
    #         look_ahead_results.append(results[term_name][k]['accuracy'])
    #
    #     df_term_column = pd.DataFrame({f'{k}': look_ahead_results})
    #     df_results = df_results.join(df_term_column)

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
    summary_mean_table = __html_table_from_dataframe(summary_df)

    standard_deviations = {
        'Summary': f'<b>Standard deviation</b>'}
    for prediction_length in df_results.columns[1:]:
        standard_deviations[prediction_length] = stdev(df_results[prediction_length])
    summary_df = pd.DataFrame(standard_deviations, index=[0])

    trimmed_standard_deviations = {
        'Summary': f'<b>Trimmed ({trimmed_proportion_to_cut * 100.0:.0f}% cut) standard deviation</b>'}
    for prediction_length in df_results.columns[1:]:
        trimmed_data = trim_proportion(df_results[prediction_length], trimmed_proportion_to_cut)
        trimmed_standard_deviations[prediction_length] = stdev(trimmed_data)
    summary_df = summary_df.append(trimmed_standard_deviations, ignore_index=True)

    summary_sd_table = __html_table_from_dataframe(summary_df, highlight_max=False)

    return summary_mean_table + summary_sd_table


def prediction_as_graphs(data, smoothed_data,term_ngrams, lims, results):
    k_range = list(next(iter(results.values())).keys())

    html_string = '''

        <table>
          <tr>
            <td>term</td>
            <td style="text-align:center">Historical counts</td>'''
    for k in k_range:
        html_string += f'            <td style="text-align:center">Predicted<br/>counts (k={k})</td>\n'
        html_string += f'            <td style="text-align:center">Predicted<br/>derivative (k={k})</td>\n'
    html_string += '''
           </tr>
'''

    for term in tqdm(results, desc='Producing graphs', unit='term'):
        html_string += f'''
          <tr>
            <td>{term}</td>
    '''
        term_index = term_ngrams.index(term)
        max_y = max(data[term_index]) * 1.2
        for k in k_range:
            max_y = max(max(results[term][k][0]['predicted_values']) * 1.2, max_y)

        fig = plt.figure(term, figsize=(6, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        print(data[term_index])
        ax.plot(data[term_index], color='b', linestyle='-', marker='x', label='Ground truth')
        if smoothed_data is not None:
            ax.plot(list(smoothed_data[term_index]), color='g', linestyle='-', marker='*',
                    label='Smoothed Ground truth')
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_ylim([0, max_y])
        ax.axvline(x=lims[0], color='k', linestyle='--')
        ax.axvline(x=lims[1], color='k', linestyle='--')

        html_string += '        <td>' + plot_to_html_image(plt) + '</td>\n'

        for k in k_range:
            last_entry = max([x for x in results[term][k].keys() if type(x) is int])
            fig = plt.figure(term, figsize=(1, 1.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(results[term][k][last_entry]['predicted_values'], color='r', linestyle='-', marker='+',
                    label='Prediction')
            ax.set_ylim([0, max_y])
            ax.set_yticklabels([])
            html_string += '            <td>' + plot_to_html_image(plt) + '</td>\n'

            fig = plt.figure(term, figsize=(1, 1.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(results[term][k][last_entry]['predicted_derivative'], color='r', linestyle='-', marker='+',
                    label='Prediction')
            # ax.set_ylim([0, max_y])
            # ax.set_yticklabels([])
            html_string += '            <td>' + plot_to_html_image(plt) + '</td>\n'

        html_string += '          </tr>\n'
    html_string += '        </table>\n'

    return html_string
