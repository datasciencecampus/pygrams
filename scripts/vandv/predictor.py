import os
import pickle

from tqdm import tqdm

from scripts.algorithms.predictor_factory import PredictorFactory as factory
from scripts.utils.Utils_emtech import timeseries_weekly_to_quarterly
from scripts.vandv.graphs import trim_leading_zero_counts


def evaluate_prediction(term_counts_per_week, term_ngrams, predictor_names, weekly_iso_dates, output_folder, test_terms,
                        prefix=None, suffix=None,
                        test_forecasts=False, normalised=False, number_of_patents_per_week=None,
                        num_prediction_periods=5):
    # TODO: maybe do that before pickling if this is the only place it is used!
    term_counts_per_week_csc = term_counts_per_week.tocsc()
    output_str = 'prediction_results_test' if test_forecasts else 'prediction_results'

    if prefix:
        output_str = prefix + '=' + output_str

    if suffix:
        output_str = output_str + '=' + suffix

    base_file_name = os.path.join(output_folder, output_str)

    if normalised:
        base_file_name += '_normalised'
    pickle_file_name = base_file_name + '_cache.pkl'

    if os.path.isfile(pickle_file_name):
        with open(pickle_file_name, 'rb') as f:
            results = pickle.load(f)
    else:
        results = {}

    training_values = {}
    test_values = {}
    test_offset = num_prediction_periods if test_forecasts else 0

    if normalised:
        quarterly_patent_dates, quarterly_patent_counts = timeseries_weekly_to_quarterly(weekly_iso_dates,
                                                                                         number_of_patents_per_week)

    for test_term in test_terms:
        term_index = term_ngrams.index(test_term)
        weekly_values = term_counts_per_week_csc.getcol(term_index).todense().ravel().tolist()[0]

        quarterly_dates, quarterly_values = timeseries_weekly_to_quarterly(weekly_iso_dates, weekly_values)

        if normalised:
            quarterly_values = [v / c for v, c in zip(quarterly_values, quarterly_patent_counts)]

        trimmed_quarterly_dates, trimmed_quarterly_int_values = trim_leading_zero_counts(quarterly_dates,
                                                                                         quarterly_values)
        trimmed_quarterly_values = [float(v) for v in trimmed_quarterly_int_values]

        term = term_ngrams[term_index]
        training_values[term] = trimmed_quarterly_values[:-test_offset - 1]
        if test_forecasts:
            test_values[term] = trimmed_quarterly_values[-test_offset - 1:-1]

    if normalised:
        term = '__ number of patents'
        test_terms = [term] + test_terms
        training_values[term] = [float(x) for x in quarterly_patent_counts[:-num_prediction_periods - 1]]
        if test_forecasts:
            test_values[term] = [float(x) for x in quarterly_patent_counts[-num_prediction_periods - 1:-1]]

    for predictor_name in predictor_names:
        if predictor_name not in results:
            results[predictor_name] = {}

        for test_term in tqdm(test_terms, unit='term', desc='Validating prediction with ' + predictor_name):
            if test_term in results[predictor_name]:
                continue

            model = factory.predictor_factory(predictor_name, test_term, training_values[test_term],
                                              num_prediction_periods)
            predicted_values = model.predict_counts()

            results[predictor_name][test_term] = (
                None, model.configuration, predicted_values, len(training_values))

        # Save after each iteration in case we abort - its very slow!
        os.makedirs(output_folder, exist_ok=True)
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(results, f)

    return results, training_values, test_values
