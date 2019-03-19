from tqdm import tqdm

from scripts.algorithms.predictor_factory import PredictorFactory as factory
from scripts.utils.utils import timeseries_weekly_to_quarterly
from scripts.vandv.graphs import trim_leading_zero_counts


def evaluate_prediction(term_counts_per_week, term_ngrams, predictor_names, weekly_iso_dates, test_terms,
                        test_forecasts=False, normalised=False, number_of_patents_per_week=None,
                        num_prediction_periods=5):
    # TODO: maybe do that before pickling if this is the only place it is used!
    term_counts_per_week_csc = term_counts_per_week.tocsc()

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
        results[predictor_name] = {}

        for test_term in tqdm(test_terms, unit='term', desc='Validating prediction with ' + predictor_name):

            model = factory.predictor_factory(predictor_name, test_term, training_values[test_term],
                                              num_prediction_periods)
            predicted_values = model.predict_counts()

            results[predictor_name][test_term] = (
                None, model.configuration, predicted_values, len(training_values))

    return results, training_values, test_values
