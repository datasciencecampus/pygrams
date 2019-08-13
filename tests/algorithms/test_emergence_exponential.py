import unittest
import numpy as np

from scripts.algorithms.emergence import Emergence


class ExponentialEmergenceTests(unittest.TestCase):
    '''
        escore=1 all yearly_values in the last year
        escore=2/3 yearly_values linearly increase from zero over 3 years (7/15 over 6 years, 0.5 infinite years)
        escore=0 yearly_values equally spread over all years (horizontal line)
        escore=-2/3 yearly_values linearly decrease to zero over 3 years (-7/15 over 6 years, -0.5 infinite years)
        escore=-1 all yearly_values in the first year
    '''

    def setUp(self):
        self.places = 5
        self.weeks_in_year = 52  # 52.1775
        num_whole_years = 6
        self.weeks = []
        for year in range(num_whole_years):
            self.weeks.append(int((year + 1) * self.weeks_in_year) - int((year) * self.weeks_in_year))
        pass

    def test_emergent_1(self):
        # Arrange
        weekly_values = [10] * 10 + [0] * self.weeks_in_year + [0] * self.weeks_in_year + [3] * self.weeks_in_year
        # yearly_weights = [0, 1, 2]
        # weekly_values = [10] * 10 + sum(np.multiply(yearly_weights * self.weeks[:len(yearly_weights)]))
        escore_expected = 1
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_2_3(self):
        # Arrange
        weekly_values = [10] * 10 + [0] * self.weeks_in_year + [1] * self.weeks_in_year + [2] * self.weeks_in_year
        escore_expected = 2 / 3
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_0(self):
        # Arrange
        weekly_values = [10] * 10 + [1] * self.weeks_in_year + [1] * self.weeks_in_year + [1] * self.weeks_in_year
        escore_expected = 0
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_neg_2_3(self):
        # Arrange
        weekly_values = [10] * 10 + [2] * self.weeks_in_year + [1] * self.weeks_in_year + [0] * self.weeks_in_year
        escore_expected = -2 / 3
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_neg_1(self):
        # Arrange
        weekly_values = [10] * 10 + [3] * self.weeks_in_year + [0] * self.weeks_in_year + [0] * self.weeks_in_year
        escore_expected = -1
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_7_15(self):
        # Arrange
        weekly_values = [10] * 10 + [0] * self.weeks_in_year + [1] * self.weeks_in_year + [2] * self.weeks_in_year + [3] * self.weeks_in_year + [4] * self.weeks_in_year + [5] * self.weeks_in_year + [5]
        escore_expected = 7 / 15
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_neg_7_15(self):
        # Arrange
        weekly_values = [10] * 10 + [5] * self.weeks_in_year + [4] * self.weeks_in_year + [3] * self.weeks_in_year + [2] * self.weeks_in_year + [1] * self.weeks_in_year + [0] * self.weeks_in_year + [0]
        escore_expected = -7 / 15
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)
