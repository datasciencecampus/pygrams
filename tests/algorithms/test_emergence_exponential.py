import unittest

from pygrams.algorithms.emergence import Emergence


class ExponentialEmergenceTests(unittest.TestCase):
    '''
        escore=1 all yearly_values in the last year
        escore=2/3 yearly_values linearly increase from zero over 3 years (7/15 over 6 years, 0.5 infinite years)
        escore=0 yearly_values equally spread over all years (horizontal line)
        escore=-2/3 yearly_values linearly decrease to zero over 3 years (-7/15 over 6 years, -0.5 infinite years)
        escore=-1 all yearly_values in the first year
    '''

    def setUp(self):
        self.weeks = 52
        self.places = 5

    def test_emergent_1(self):
        # Arrange
        weekly_values = [10] * 10 + [0] * self.weeks + [0] * self.weeks + [3] * self.weeks
        escore_expected = 1
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_2_3(self):
        # Arrange
        weekly_values = [10] * 10 + [0] * self.weeks + [1] * self.weeks + [2] * self.weeks
        escore_expected = 2 / 3
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_0(self):
        # Arrange
        weekly_values = [10] * 10 + [1] * self.weeks + [1] * self.weeks + [1] * self.weeks
        escore_expected = 0
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_neg_2_3(self):
        # Arrange
        weekly_values = [10] * 10 + [2] * self.weeks + [1] * self.weeks + [0] * self.weeks
        escore_expected = -2 / 3
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_neg_1(self):
        # Arrange
        weekly_values = [10] * 10 + [3] * self.weeks + [0] * self.weeks + [0] * self.weeks
        escore_expected = -1
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_7_15(self):
        # Arrange (add an extra leading 0 for the first 53 week year)
        weekly_values = [10] * 10 + [0] + [0] * self.weeks + [1] * self.weeks + [2] * self.weeks + [3] * self.weeks\
                        + [4] * self.weeks + [5] * self.weeks
        escore_expected = 7 / 15
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)

    def test_emergent_neg_7_15(self):
        # Arrange (add an extra leading 0 for the first 53 week year)
        weekly_values = [10] * 10 + [0] + [5] * self.weeks + [4] * self.weeks + [3] * self.weeks + [2] * self.weeks \
                        + [1] * self.weeks + [0] * self.weeks
        escore_expected = -7 / 15
        # Act
        escore_actual = Emergence.escore_exponential(weekly_values)
        # Assert
        self.assertAlmostEqual(escore_expected, escore_actual, places=self.places)
