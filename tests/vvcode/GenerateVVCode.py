from random import randint

import pandas as pd

df = pd.read_pickle("data/US-1000-random.pkl.bz2")
df = df.reset_index(drop=True)

indices = [randint(0, 999) for p in range(100)]

rand_df = df.ix[indices]

for index, row in rand_df.iterrows():
    pid = row['patent_id']
    abstract = row['abstract']

    testdef = '    def test_patent_x' + pid + '(self):'
    testbody = '        text = ' + '\'\'\'' + abstract + '\'\'\''

    testrest = '''        expected = []
        actual = self.key_term_extractor.extract(text)

        self.assertGreaterOrEqualDiceScore(expected, actual)'''

    print(testdef)
    print(testbody)
    print(testrest)
    print("")
