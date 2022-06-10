import os


def us_patents_random_pickle_name_gen(n):
    return os.path.join('data', f'USPTO-random-{n}.pkl.bz2')


us_patents_random_100_pickle_name = us_patents_random_pickle_name_gen(100)
us_patents_random_1000_pickle_name = us_patents_random_pickle_name_gen(1000)
us_patents_random_10000_pickle_name = us_patents_random_pickle_name_gen(10000)
us_patents_random_100000_pickle_name = us_patents_random_pickle_name_gen(100000)

us_patents_citation_dictionary_1of2_pickle_name = os.path.join('data', 'USPTO-citation-dictionary-family-pt1.pkl.bz2')
us_patents_citation_dictionary_2of2_pickle_name = os.path.join('data', 'USPTO-citation-dictionary-family-pt2.pkl.bz2')

global_stopwords_filename = os.path.join('config', 'stopwords_glob.txt')
ngram_stopwords_filename = os.path.join('config', 'stopwords_n.txt')
unigram_stopwords_filename = os.path.join('config', 'stopwords_uni.txt')
