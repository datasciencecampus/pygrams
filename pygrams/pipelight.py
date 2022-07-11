from math import log10, floor

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm

import pygrams.data_factory as data_factory
from pygrams.utils import utils
from pygrams.utils.date_utils import generate_year_week_dates
from pygrams.tfidf_reduce import TfidfReduce
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from pygrams.text_processing import StemTokenizer, lowercase_strip_accents_and_ownership, WordAnalyzer
import matplotlib.pyplot as plt


class Pipelight(object):
    def __init__(self, data_filename, docs_mask_dict, pick_method='sum', ngram_range=(1, 3), text_header='abstract',
                 max_df=0.3, prefilter_terms=0, output_name=',', emergence_index='porter'):

        self.__emergence_index = emergence_index

        # load data
        self.__data_filename = data_filename
        self.__date_dict = docs_mask_dict['date']
        self.__timeseries_date_dict = docs_mask_dict['timeseries_date']
        self.__timeseries_data = []
        self.__timeseries_outputs=None

        self.__emergence_list = []
        self.__pick_method = pick_method

        # get data
        dataframe = data_factory.get(data_filename)
        utils.remove_empty_documents(dataframe, text_header)

        pg_text = dataframe[text_header]
        dates = dataframe[docs_mask_dict['date_header']]
        self.__dates = generate_year_week_dates(dataframe, docs_mask_dict['date_header'])
        stop_from_file=['first', 'plurality', 'etc']
        my_stop_words = ENGLISH_STOP_WORDS.union(stop_from_file)

        # calculate tfidf

        tokenizer = StemTokenizer()

        vectorizer = CountVectorizer(
            max_df=max_df,
            min_df=floor(log10(dataframe.shape[0])),
            ngram_range=ngram_range,
            tokenizer=tokenizer,
            strip_accents='ascii',
            stop_words=my_stop_words
        )
        count_matrix = vectorizer.fit_transform(pg_text)

        tfidf_transformer = TfidfTransformer(smooth_idf=False, norm=None)
        tfidf = tfidf_transformer.fit_transform(count_matrix)

        # feature selection

        tfidf_reduce_obj = TfidfReduce(tfidf, vectorizer.vocabulary_)
        term_score_mp = tfidf_reduce_obj.extract_ngrams_from_docset('mean_prob')
        num_tuples_to_retain = min(prefilter_terms, len(term_score_mp))

        term_score_entropy = tfidf_reduce_obj.extract_ngrams_from_docset('entropy')
        term_score_variance = tfidf_reduce_obj.extract_ngrams_from_docset('variance')

        feature_subset_mp = sorted([x[1] for x in term_score_mp[:num_tuples_to_retain]])
        feature_subset_variance = sorted([x[1] for x in term_score_variance[:num_tuples_to_retain]])
        feature_subset_entropy = sorted([x[1] for x in term_score_entropy[:num_tuples_to_retain]])

        feature_subset = list(set(feature_subset_mp).union(set(feature_subset_variance)).union(feature_subset_entropy))

        tfidf.data = np.ones(len(tfidf.data))
        sdf = pd.DataFrame.sparse.from_spmatrix(tfidf, columns=vectorizer.vocabulary_)[feature_subset]
        sdf = sdf.set_index(pd.Index(dates.tolist()))

        timeseries_df = sdf.resample('Q').sum()
        # sdf_all = sdf.sum(axis=1)
        # sdf_all.resample('Q').sum()

        timeseries={}
        smooth_timeseries={}

        for term in tqdm(feature_subset, unit='term',
                               desc='Calculating quarterly timeseries',
                               leave=False, unit_scale=True):

            smooth_series = savgol_filter(timeseries_df[term], 9, 2, mode='nearest')
            smooth_series_no_negatives = np.clip(smooth_series, a_min=0, a_max=None)
            smooth_timeseries[term] = smooth_series_no_negatives

        # net growth
        emergence = {}
        for term in tqdm(feature_subset, unit='term',
                         desc='Calculating quarterly timeseries',
                         leave=False, unit_scale=True):
            term_series = smooth_timeseries[term][-10:]
            emergence[term] = sum([(term_series[i] - term_series[i-1])/y if y > 1.0 and i >0 else 0.0 for i, y in enumerate(term_series)])
        emergence = {k: emergence[k] for k in sorted(emergence, key=emergence.get)}
        print(emergence)
        timeseries_df[list(emergence.keys())[0:10]].plot(subplots=True, figsize=(8, 8))
        plt.legend(loc='best')
        plt.show()

