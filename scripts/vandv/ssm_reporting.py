import pandas as pd
from scipy.stats import trim_mean


def __html_table_from_dataframe(df_results, terms, term_style='{:.0}'):
    df_summary_table = df_results.style.hide_index()
    df_summary_table = df_summary_table.set_table_styles([
        dict(selector='table', props=[('border-collapse', 'collapse')]),
        dict(selector='td', props=[('border', '2px solid black'),
                                   ('text-align', 'right'),
                                   ('padding-left', '15px'),
                                   ('padding-right', '15px')])
    ])
    for term in terms:
        df_summary_table = df_summary_table.format({term: term_style})
    df_summary_table = df_summary_table.highlight_max(axis=1)

    heading = '<style type="text/css">table {border-collapse: collapse;} </style>\n'
    return heading + df_summary_table.render()


def html_table(results, prediction_lengths):
    df_results = pd.DataFrame({'terms': list(results.keys())})
    for prediction_length in prediction_lengths:
        prediction_length_results = []
        for term_name in results:
            prediction_length_results.append(results[term_name][prediction_length])

        df_term_column = pd.DataFrame({f'{prediction_length}': prediction_length_results})
        df_results = df_results.join(df_term_column)

    results_table = __html_table_from_dataframe(df_results, 'SSM')

    trimmed_mean_proportion_to_cut = 0.1
    trimmed_means = {
        'terms': f'<b>Trimmed ({trimmed_mean_proportion_to_cut * 100.0:.0f}% cut) <br/> mean</b>'}
    for prediction_length in prediction_lengths:
        trimmed_means[f'{prediction_length}'] = trim_mean(df_results[f'{prediction_length}'],
                                                          trimmed_mean_proportion_to_cut)
    # print(f'{term_name} trimmed mean={trimmed_means:.1f}')
    df_results = df_results.append(trimmed_means, ignore_index=True)

    # standard_deviations = {
    #     'terms': f'<b>Standard deviation <br/>of {metric_name}</b>'}
    # for predictor_name in predictor_names:
    #     predictor_display_name = predictor_name.replace('-', '<br/>')
    #     results_without_nan = [x for x in df_results[predictor_display_name] if not math.isnan(x)]
    #     standard_deviations[predictor_display_name] = stdev(results_without_nan)
    #     print(f'{predictor_name} {metric_name} standard deviation={standard_deviations[predictor_display_name]:.1f}')
    # df_results = df_results.append(standard_deviations, ignore_index=True)

    summary_df = pd.DataFrame(trimmed_means, index=[0])
    summary_table = __html_table_from_dataframe(summary_df, 'SSM summary')

    # summary_df = summary_df.append(standard_deviations, ignore_index=True)

    return f'<h2>State Space Model Results</h2>\n{results_table}<p/>{summary_table}\n'
