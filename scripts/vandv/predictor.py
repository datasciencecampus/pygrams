from tqdm import tqdm

from scripts.algorithms.predictor_factory import PredictorFactory as factory
from scripts.utils.date_utils import timeseries_weekly_to_quarterly


# TODO quarterly values, should become timeseries
def evaluate_prediction(timeseries_terms, term_ngrams, predictor_names, test_terms, test_forecasts=False,
                        timeseries_all=None, num_prediction_periods=5, smoothed_series=None):
    # TODO: maybe do that before pickling if this is the only place it is used!

    results = {}
    smoothed_training_values = {}
    smoothed_test_values = {}
    training_values = {}
    test_values = {}
    test_offset = num_prediction_periods if test_forecasts else 0

    for test_term in test_terms:
        term_index = term_ngrams.index(test_term)
        timeseries_term = timeseries_terms[term_index]
        if timeseries_all is not None:
            timeseries_term = [v / c for v, c in zip(timeseries_term, timeseries_all)]

        timeseries_term_float = [float(v) for v in timeseries_term]

        term = term_ngrams[term_index]
        training_values[term] = timeseries_term_float[:-test_offset - 1]
        if smoothed_series is not None:
            smoothed_training_values[term]=smoothed_series[term_index][:-test_offset - 1]
        if test_forecasts:
            test_values[term] = timeseries_term_float[-test_offset - 1:-1]
            smoothed_test_values[term] = smoothed_series[term_index][-test_offset - 1:-1]

    if timeseries_all is not None:
        term = '__ number of patents'
        test_terms = [term] + test_terms
        training_values[term] = [float(x) for x in timeseries_all[:-num_prediction_periods - 1]]
        if test_forecasts:
            test_values[term] = [float(x) for x in timeseries_all[-num_prediction_periods - 1:-1]]

    for predictor_name in predictor_names:
        results[predictor_name] = {}

        for test_term in tqdm(test_terms, unit='term', desc='Validating prediction with ' + predictor_name):

            model = factory.predictor_factory(predictor_name, test_term, training_values[test_term],
                                              num_prediction_periods)
            predicted_values = model.predict_counts()

            results[predictor_name][test_term] = (
                None, model.configuration, predicted_values, len(training_values))

    return results, training_values, test_values, smoothed_training_values, smoothed_test_values
