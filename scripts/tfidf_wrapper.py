from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scripts.text_processing import StemTokenizer, lowercase_strip_accents_and_ownership, WordAnalyzer
from scripts.tfidf_reduce import TfidfReduce


class TFIDF:

    def __init__(self, text_series, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer()):
        WordAnalyzer.init(
            tokenizer=tokenizer,
            preprocess=lowercase_strip_accents_and_ownership,
            ngram_range=ngram_range)

        self.__vectorizer = CountVectorizer(
            max_df=max_document_frequency,
            min_df=1,
            ngram_range=ngram_range,
            analyzer=WordAnalyzer.analyzer
        )

        self.__ngram_counts = self.__vectorizer.fit_transform(text_series)
        self.__feature_names = self.__vectorizer.get_feature_names()

        self.__tfidf_transformer = TfidfTransformer(smooth_idf=False)
        self.__tfidf_matrix = self.__tfidf_transformer.fit_transform(self.__ngram_counts)

        self.__tfidf_reduce_obj = TfidfReduce(self.__tfidf_matrix, self.__feature_names)
        self.__term_score_tuples = self.__tfidf_reduce_obj.extract_ngrams_from_docset('sum')
        features = sorted([x[1] for x in self.__term_score_tuples[:len(self.__term_score_tuples)//20]])
        indices = sorted([self.__vectorizer.vocabulary_.get(x) for x in features])

        self.__new_tfidf = self.__tfidf_matrix[:, indices]
        self.__new_idf   = self.__tfidf_transformer.idf_[indices]
        self.__new_vocab = {}
        for i in range (len(features)):
            self.__new_vocab.setdefault(features[i], i)
        self.__new_features = features

    @property
    def idf(self):
        return self.__new_idf

    @property
    def tfidf_matrix(self):
        return self.__new_tfidf

    @property
    def vocabulary(self):
        return self.__new_vocab

    @property
    def feature_names(self):
        return self.__new_features
