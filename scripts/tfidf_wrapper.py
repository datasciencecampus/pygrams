import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scripts.text_processing import StemTokenizer, lowercase_strip_accents_and_ownership, WordAnalyzer
from scripts.utils import utils


def tfidf_from_text(text_series, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer(), min_df=1):
    WordAnalyzer.init(
        tokenizer=tokenizer,
        preprocess=lowercase_strip_accents_and_ownership,
        ngram_range=ngram_range)

    #TODO add option to increase uint8 to uint16 or 32 on user
    vectorizer = CountVectorizer(
        max_df=max_document_frequency,
        min_df=min_df,
        ngram_range=ngram_range,
        analyzer=WordAnalyzer.analyzer,
        dtype=np.uint8
    )

    count_matrix = vectorizer.fit_transform(text_series)
    feature_names = vectorizer.get_feature_names()

    return _TFIDF(count_matrix, vectorizer.vocabulary_, feature_names)


def tfidf_subset_from_features(tfidf_obj, feature_subset):
    l2 = tfidf_obj.l2_norm
    indices = sorted([tfidf_obj.vocabulary.get(x) for x in feature_subset])

    new_count_matrix = tfidf_obj.count_matrix[:, indices]
    new_vocabulary = {}
    for i in range(len(feature_subset)):
        new_vocabulary.setdefault(feature_subset[i], i)

    return _TFIDF(new_count_matrix, new_vocabulary, feature_subset, l2_norm=l2)


class _TFIDF:

    def __init__(self,  count_matrix, vocabulary, feature_names, l2_norm=None):
        self.__l2_norm = l2_norm
        self.__count_matrix = count_matrix
        self.__vocabulary = vocabulary
        self.__feature_names = feature_names
        self.__tfidf_transformer = None
        self.__tfidf_matrix=None

    def __trigger_transformer(self):
        self.__tfidf_transformer = TfidfTransformer(smooth_idf=False, norm=None)
        self.__tfidf_matrix = self.__tfidf_transformer.fit_transform(self.__count_matrix)
        if self.__l2_norm is None:
            self.__l2_norm = utils.l2normvec(self.__tfidf_matrix)
        self.__tfidf_matrix = utils.apply_l2normvec(self.__tfidf_matrix, self.__l2_norm)

    def apply_weights(self, weights_matrix):
        self.__count_matrix = self.__count_matrix.multiply(weights_matrix)
        self.__tfidf_matrix = self.__tfidf_matrix.multiply(weights_matrix)
        self.__count_matrix.data = np.array([np.uint8(round(x)) for x in self.__count_matrix.data])

    @property
    def l2_norm(self):
        return self.__l2_norm

    @property
    def idf(self):
        if self.__tfidf_transformer is None:
            self.__trigger_transformer()
        return self.__tfidf_transformer.idf_

    @property
    def count_matrix(self):
        return self.__count_matrix

    @property
    def tfidf_matrix(self):
        if self.__tfidf_transformer is None:
            self.__trigger_transformer()
        return self.__tfidf_matrix

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def feature_names(self):
        return self.__feature_names
