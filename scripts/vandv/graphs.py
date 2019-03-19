# from sys import platform as sys_pf
# if sys_pf == 'darwin':
#     import matplotlib
#     matplotlib.use("TkAgg")

import base64
import math
from io import BytesIO
from math import sqrt
from statistics import stdev

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import trim_mean
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

from scripts.utils.utils import iso_to_gregorian


def trim_leading_zero_counts(dates, values):
    first_non_zero_index = None
    for index, value in enumerate(values):
        if value != 0.0:
            first_non_zero_index = index
            break

    if first_non_zero_index is None:
        return [], []

    return dates[first_non_zero_index:], values[first_non_zero_index:]


def plot_to_html_image(plt):
    img = BytesIO()
    plt.savefig(img, transparent=True, bbox_inches='tight')
    img.seek(0)
    plt.close()
    html_string = '<img src="data:image/png;base64,' + base64.b64encode(img.read()).decode("UTF-8") + '"/>'
    return html_string


def extract_rmse(predicted_values, test_term, test_values, training_values):
    if all(not math.isnan(x) for x in predicted_values):
        max_y = max([max(training_values[test_term]), max(test_values[test_term])])
        y_scale = 100.0 / max_y
        mse = mean_squared_error([t * y_scale for t in test_values[test_term]],
                                 [p * y_scale for p in predicted_values])
        rmse = sqrt(mse)
    else:
        rmse = math.nan
    return rmse


def extract_mean_absolute_error(predicted_values, test_term, test_values, _):
    if all(not math.isnan(x) for x in predicted_values):
        mae = mean_absolute_error(test_values[test_term], predicted_values)
    else:
        mae = math.nan
    return mae


def extract_relative_rmse(predicted_values, test_term, test_values, training_values):
    max_training_value = max(training_values[test_term])

    if all(not math.isnan(x) for x in predicted_values):
        mse = mean_squared_error(test_values[test_term], predicted_values)
        relative_rmse = sqrt(mse) / max_training_value
    else:
        relative_rmse = math.nan
    return relative_rmse


def html_table(df_results, predictor_names, predictor_style):
    df_summary_table = df_results.style.hide_index()
    df_summary_table = df_summary_table.set_table_styles([
        dict(selector='table', props=[('border-collapse', 'collapse')]),
        dict(selector='td', props=[('border', '2px solid black'),
                                   ('text-align', 'right'),
                                   ('padding-left', '15px'),
                                   ('padding-right', '15px')])
    ])
    for predictor_name in predictor_names:
        df_summary_table = df_summary_table.format({predictor_name: predictor_style})
    df_summary_table = df_summary_table.highlight_min(axis=1)

    heading = '<style type="text/css">table {border-collapse: collapse;} </style>\n'
    return heading + df_summary_table.render()


def create_html_table(predictor_names, results, test_terms, test_values, training_values, metric_name, value_extractor,
                      table_term_format='{:.1f}', value_scale=1.0):
    if value_scale == 1.0:
        scale_description = ''
    else:
        scale_description = f' (results scaled by a factor of {value_scale:,}, except number of patents)'

    print()
    print(metric_name)

    df_results = pd.DataFrame({'terms': test_terms})
    predictor_display_names = []
    for predictor_name in predictor_names:
        term_results = []
        for test_term in test_terms:
            result = results[predictor_name][test_term]

            v = value_extractor(result[2], test_term, test_values, training_values)

            if not test_term.startswith('__'):
                v *= value_scale

            term_results.append(v)

        predictor_display_name = predictor_name.replace('-', '<br/>')
        predictor_display_names.append(predictor_display_name)

        df_term_column = pd.DataFrame({predictor_display_name: term_results})
        df_results = df_results.join(df_term_column)

    trimmed_mean_proportion_to_cut = 0.1
    trimmed_means = {
        'terms': f'<b>Trimmed ({trimmed_mean_proportion_to_cut * 100.0:.0f}% cut) <br/> mean of {metric_name}</b>'}
    for predictor_name in predictor_names:
        predictor_display_name = predictor_name.replace('-', '<br/>')
        trimmed_means[predictor_display_name] = trim_mean(df_results[predictor_display_name],
                                                          trimmed_mean_proportion_to_cut)
        print(f'{predictor_name} {metric_name} trimmed mean={trimmed_means[predictor_display_name]:.1f}')
    df_results = df_results.append(trimmed_means, ignore_index=True)

    standard_deviations = {
        'terms': f'<b>Standard deviation <br/>of {metric_name}</b>'}
    for predictor_name in predictor_names:
        predictor_display_name = predictor_name.replace('-', '<br/>')
        results_without_nan = [x for x in df_results[predictor_display_name] if not math.isnan(x)]
        standard_deviations[predictor_display_name] = stdev(results_without_nan)
        print(f'{predictor_name} {metric_name} standard deviation={standard_deviations[predictor_display_name]:.1f}')
    df_results = df_results.append(standard_deviations, ignore_index=True)

    results_table = html_table(df_results, predictor_display_names, table_term_format)

    summary_df = pd.DataFrame(trimmed_means, index=[0])
    summary_df = summary_df.append(standard_deviations, ignore_index=True)

    return f'<h2>{metric_name}{scale_description}</h2>\n{results_table}<p/>\n', summary_df


def graphs_of_predicted_term_counts(predictor_names, results, test_terms, training_values, test_values=None,
                                    normalised=False):
    html_string = '''

    <p/>
    <h2>Graphs of predicted term counts</h2>
    <table>
      <tr>
        <td>term</td>
        <td style="text-align:center">Historical data</td>
'''
    for predictor_name in predictor_names:
        html_string += f'''        <td style="text-align:center">{predictor_name}</td>\n'''
    html_string += '      </tr>\n'
    for test_term in tqdm(test_terms, desc='Producing graphs', unit='term'):
        html_string += f'''
      <tr>
        <td>{test_term}</td>
'''
        if test_values is not None and len(test_values) > 0:
            max_y = max([max(training_values[test_term]), max(test_values[test_term])]) * 1.2
        else:
            max_y = max(training_values[test_term]) * 1.2

        fig = plt.figure(test_term, figsize=(6, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(training_values[test_term], color='b', linestyle='-', marker='x', label='Ground truth')
        ax.set_ylabel('Normalised\nFrequency' if normalised else 'Frequency', fontsize=12)
        ax.set_ylim([0, max_y])

        html_string += '        <td>' + plot_to_html_image(plt) + '</td>\n'

        for predictor_name in predictor_names:
            result = results[predictor_name][test_term]

            fig = plt.figure(test_term, figsize=(1, 1.5), dpi=100)
            ax = fig.add_subplot(111)
            if test_values is not None and len(test_values) > 0:
                ax.plot(test_values[test_term], color='b', linestyle='-', marker='x', label='Ground truth')

            ax.plot(result[2], color='r', linestyle='-', marker='+', label='Prediction')
            ax.set_ylim([0, max_y])
            ax.set_yticklabels([])

            html_string += '        <td>' + plot_to_html_image(plt) + '</td>\n'
        html_string += '    </tr>\n'
    html_string += '  </table>\n'
    return html_string


def report_prediction_as_graphs_html(results, predictor_names, weekly_iso_dates,
                                     test_values, test_terms, training_values, test_forecasts=False, normalised=False):
    html_string = f'''    <h2>Graphs</h1>
'''

    if normalised:
        test_terms = ['__ number of patents'] + list(test_terms)

    first_patent_date = iso_to_gregorian(weekly_iso_dates[0])
    last_patent_date = iso_to_gregorian(weekly_iso_dates[-1])
    html_string += f'Patents from {first_patent_date:%d %B %Y} to {last_patent_date:%d %B %Y}.<p/>\n'

    if test_forecasts:
        html_string += 'Note that any NaN results will be ignored when calculating standard deviation<p/>\n'

        rmse_html_string, rmse_df = create_html_table(predictor_names, results, test_terms, test_values,
                                                      training_values,
                                                      'Relative RMSE', extract_relative_rmse,
                                                      table_term_format='{:.1%}')
        if normalised:
            abs_error_value_scale = 10000.0
        else:
            abs_error_value_scale = 1.0
        abs_error_html_string, abs_error_df = create_html_table(predictor_names, results, test_terms, test_values,
                                                                training_values,
                                                                'Absolute error', extract_mean_absolute_error,
                                                                value_scale=abs_error_value_scale)

        avg_rmse_html_string, avg_rmse_df = create_html_table(predictor_names, results, test_terms, test_values,
                                                              training_values, 'Average RMSE', extract_rmse)

        predictor_display_names = []
        for predictor_name in predictor_names:
            predictor_display_name = predictor_name.replace('-', '<br/>')
            predictor_display_names.append(predictor_display_name)

        rmse_table = html_table(rmse_df, predictor_display_names, '{:.1%}')
        abs_error_table = html_table(abs_error_df, predictor_display_names, '{:.1f}')
        avg_rmse_table = html_table(avg_rmse_df, predictor_display_names, '{:.1f}')

        html_string += f'<h2>Summary</h2>\n{rmse_table}<p/>\n{abs_error_table}<p/>\n{avg_rmse_table}<p/>\n'

        html_string += rmse_html_string + abs_error_html_string + avg_rmse_html_string

    html_string += graphs_of_predicted_term_counts(predictor_names, results, test_terms, training_values,
                                                   test_values=test_values, normalised=normalised)

    return html_string
