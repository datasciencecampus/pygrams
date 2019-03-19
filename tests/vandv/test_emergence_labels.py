import unittest

import numpy as np

from scripts.vandv.emergence_labels import map_prediction_to_emergence_label

RAPIDLY_EMERGENT = 'rapidly emergent'
EMERGENT = 'emergent'
STATIONARY = 'stationary'
DECLINING = 'declining'


class EmergenceLabelTests(unittest.TestCase):
    def test_emergence_label(self):
        actual_predictor_name = 'Actual'
        test_predictor_name = 'test predictor'
        rapidly_emergent_term_name = 'rapidly emergent term'
        emergent_term_name = 'emergent term'
        stationary_term_name = 'stationary term'
        declining_term_name = 'declining term'
        test_terms = [rapidly_emergent_term_name, emergent_term_name, stationary_term_name, declining_term_name]
        predictors_to_run = [test_predictor_name]

        num_training_values = 20
        results = {
            test_predictor_name: {
                rapidly_emergent_term_name: (None, '', np.array([1.11, 1.22, 1.33, 1.44, 1.55]), num_training_values),
                emergent_term_name: (None, '', np.array([1.021, 1.042, 1.063, 1.084, 1.106]), num_training_values),
                stationary_term_name: (None, '', np.array([0.981, 0.962, 0.943, 0.924, 0.905]), num_training_values),
                declining_term_name: (None, '', np.array([0.979, 0.958, 0.937, 0.916, 0.895]), num_training_values),
            }
        }

        training_values = {
            rapidly_emergent_term_name: [1, 1, 1, 1, 1],
            emergent_term_name: [1, 1, 1, 1, 1],
            stationary_term_name: [1, 1, 1, 1, 1],
            declining_term_name: [1, 1, 1, 1, 1],
        }

        test_values = {  # generates 'actual'
            rapidly_emergent_term_name: [1.5, 2, 7, 6, 5],
            emergent_term_name: [1.09, 1.18, 1.27, 1.36, 1.45],
            stationary_term_name: [1.019, 1.038, 1.057, 1.076, 1.095],
            declining_term_name: [0.9, 0.8, 0.7, 0.6, 0.5],
        }

        emergence_linear_thresholds = (
            (RAPIDLY_EMERGENT, 0.1),
            (EMERGENT, 0.02),
            (STATIONARY, -0.02),
            (DECLINING, None)
        )
        predicted_emergence = map_prediction_to_emergence_label(results, training_values, test_values,
                                                                predictors_to_run, test_terms,
                                                                emergence_linear_thresholds)

        self.assertEqual(RAPIDLY_EMERGENT, predicted_emergence[actual_predictor_name][rapidly_emergent_term_name])
        self.assertEqual(RAPIDLY_EMERGENT, predicted_emergence[test_predictor_name][rapidly_emergent_term_name])

        self.assertEqual(EMERGENT, predicted_emergence[actual_predictor_name][emergent_term_name])
        self.assertEqual(EMERGENT, predicted_emergence[test_predictor_name][emergent_term_name])

        self.assertEqual(STATIONARY, predicted_emergence[actual_predictor_name][stationary_term_name])
        self.assertEqual(STATIONARY, predicted_emergence[test_predictor_name][stationary_term_name])

        self.assertEqual(DECLINING, predicted_emergence[actual_predictor_name][declining_term_name])
        self.assertEqual(DECLINING, predicted_emergence[test_predictor_name][declining_term_name])
