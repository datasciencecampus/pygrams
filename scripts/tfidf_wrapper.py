from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scripts.text_processing import StemTokenizer, lowercase_strip_accents_and_ownership, WordAnalyzer


def tfidf_from_text(text_series, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer()):
    WordAnalyzer.init(
        tokenizer=tokenizer,
        preprocess=lowercase_strip_accents_and_ownership,
        ngram_range=ngram_range)

    vectorizer = CountVectorizer(
        max_df=max_document_frequency,
        min_df=1,
        ngram_range=ngram_range,
        analyzer=WordAnalyzer.analyzer
    )

    ngram_counts = vectorizer.fit_transform(text_series)
    feature_names = vectorizer.get_feature_names()

    tfidf_transformer = TfidfTransformer(smooth_idf=False)
    tfidf_matrix = tfidf_transformer.fit_transform(ngram_counts)

    return _TFIDF(tfidf_transformer.idf_, tfidf_matrix, vectorizer.vocabulary_, feature_names)


def tfidf_subset_from_features(tfidf_obj, feature_subset):
    indices = sorted([tfidf_obj.vocabulary.get(x) for x in feature_subset])

    new_tfidf_matrix = tfidf_obj.tfidf_matrix[:, indices]
    new_idf = tfidf_obj.idf[indices]
    new_vocabulary = {}
    for i in range(len(feature_subset)):
        new_vocabulary.setdefault(feature_subset[i], i)

    return _TFIDF(new_idf, new_tfidf_matrix, new_vocabulary, feature_subset)


class _TFIDF:

    def __init__(self, idf, tfidf_matrix, vocabulary, feature_names):
        self.__idf = idf
        self.__tfidf_matrix = tfidf_matrix
        self.__vocabulary = vocabulary
        self.__feature_names = feature_names

    @property
    def idf(self):
        return self.__idf

    @property
    def tfidf_matrix(self):
        return self.__tfidf_matrix

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def feature_names(self):
        return self.__feature_names
