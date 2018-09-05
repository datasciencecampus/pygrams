import pandas as pd

from tests.vvcode.abstracts2pickle import us_vv_patents_pickle_name

# Creates code template for user to add in expected terms so these can be compared against generated terms
if __name__ == '__main__':
    df = pd.read_pickle(us_vv_patents_pickle_name)
    df = df.reset_index(drop=True)

    for index, row in df.iterrows():
        abstract = row['abstract']

        test_def = f'    def test_patent_{index}(self):'
        test_body = f"        text = '''{abstract}'''"

        test_rest = '''        expected = []
            actual = self.key_term_extractor.extract(text)
    
            self.assertGreaterOrEqualDiceScore(expected, actual)'''

        print(test_def)
        print(test_body)
        print(test_rest)
        print("")
