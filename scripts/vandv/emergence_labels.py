import numpy as np
import pandas as pd
from tqdm import tqdm


def map_prediction_to_emergence_label(results, training_values, test_values, predictors_to_run, test_terms,
                                      emergence_linear_thresholds=(
                                              ('rapidly emergent', 0.1),
                                              ('emergent', 0.02),
                                              ('stationary', -0.02),
                                              ('declining', None)
                                      )):
    def __map_helper(normalised_counts_to_trend, predicted_emergence, predictor_name, test_term,
                     emergence_linear_thresholds):
        if np.isnan(sum(normalised_counts_to_trend)):
            predicted_emergence[predictor_name][test_term] = 'Fail'
            return

        x_data = range(len(normalised_counts_to_trend))

        trend = np.polyfit(x_data, normalised_counts_to_trend, 1)

        emergence = emergence_linear_thresholds[-1][0]

        for emergence_threshold in emergence_linear_thresholds[:-1]:
            if trend[0] > emergence_threshold[1]:
                emergence = emergence_threshold[0]
                break

        predicted_emergence[predictor_name][test_term] = emergence

    predicted_emergence = {}

    if test_values:
        predictor_name = 'Actual'
        predicted_emergence[predictor_name] = {}

        for test_term in tqdm(test_terms, unit='term', desc='Labelling prediction ' + predictor_name):

            counts_to_trend = test_values[test_term]
            max_training_value = max(training_values[test_term])
            normalised_counts_to_trend = [x / max_training_value for x in counts_to_trend]

            __map_helper(normalised_counts_to_trend, predicted_emergence, predictor_name, test_term,
                         emergence_linear_thresholds)

    for predictor_name in predictors_to_run:
        predicted_emergence[predictor_name] = {}

        for test_term in tqdm(test_terms, unit='term', desc='Labelling prediction ' + predictor_name):
            (none, configuration, predicted_values, num_training_values) = results[predictor_name][test_term]

            counts_to_trend = predicted_values.ravel().tolist()
            max_training_value = max(training_values[test_term])
            normalised_counts_to_trend = [x / max_training_value for x in counts_to_trend]

            __map_helper(normalised_counts_to_trend, predicted_emergence, predictor_name, test_term,
                         emergence_linear_thresholds)

    return predicted_emergence


def report_predicted_emergence_labels_html(predicted_emergence, emergence_colours={
    'highly emergent': 'lime',
    'emergent': 'green',
    'stationary': 'black',
    'declining': 'red'}):
    html_string = f'''
    <h2>Emergence Label Prediction</h1>
'''

    # df = pd.DataFrame(predicted_emergence, index=[0])
    test_terms = list(predicted_emergence[list(predicted_emergence.keys())[0]].keys())
    df_results = pd.DataFrame({'terms': test_terms})
    predictor_display_names = []
    for predictor_name in predicted_emergence:
        term_results = []
        for test_term in predicted_emergence[predictor_name]:
            result = predicted_emergence[predictor_name][test_term]

            term_results.append(result)

        predictor_display_name = predictor_name.replace('-', '<br/>')
        predictor_display_names.append(predictor_display_name)

        df_term_column = pd.DataFrame({predictor_display_name: term_results})
        df_results = df_results.join(df_term_column)

    df_summary_table = df_results.style.hide_index()
    df_summary_table = df_summary_table.set_table_styles([
        dict(selector='table', props=[('border-collapse', 'collapse')]),
        dict(selector='td', props=[('border', '2px solid black'),
                                   ('text-align', 'right'),
                                   ('padding-left', '15px'),
                                   ('padding-right', '15px')])
    ])

    def colour_emergence(val):
        colour = 'black'
        if val in emergence_colours:
            colour = emergence_colours[val]
        return f'color: {colour}'

    df_summary_table = df_summary_table.applymap(colour_emergence)

    # for predictor_name in predictor_names:
    #     df_summary_table = df_summary_table.format({predictor_name: predictor_style})
    # df_summary_table = df_summary_table.highlight_min(axis=1)

    html_string += '<style type="text/css">table {border-collapse: collapse;} </style>\n'
    html_string += df_summary_table.render()

    return html_string
