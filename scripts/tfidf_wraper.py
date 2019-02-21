import numpy as np
from scipy.sparse import csr_matrix
from scripts.text_processing import StemTokenizer, lowercase_strip_accents_and_ownership, WordAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#add pickle option
class TFIDF:

    def __init__(self, docs_df, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer(),
                 id_header='patent_id', text_header='abstract', date_header='publication_date',
                 normalize_doc_length=False, uni_factor=0.8):

        self.__dataframe = docs_df

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

        self.__id_header = id_header
        self.__text_header = text_header
        self.__date_header = date_header
        self.__uni_factor = uni_factor

        num_docs_before_sift = self.__dataframe.shape[0]
        print(self.__text_header)
        self.__dataframe.dropna(subset=[self.__text_header], inplace=True)
        num_docs_after_sift = self.__dataframe.shape[0]
        num_docs_sifted = num_docs_before_sift - num_docs_after_sift
        print(f'Dropped {num_docs_sifted:,} from {num_docs_before_sift:,} docs due to empty text field')

        self.__ngram_counts = self.__vectorizer.fit_transform(self.__dataframe[self.__text_header])
        self.__feature_names = self.__vectorizer.get_feature_names()

        if normalize_doc_length:
            self.__ngram_counts = csr_matrix(self.__ngram_counts, dtype=np.float64, copy=True)
            self.__normalize_rows()

        self.__tfidf_transformer = TfidfTransformer(smooth_idf=False)
        self.__tfidf_matrix = self.__tfidf_transformer.fit_transform(self.__ngram_counts)
        max_bi_freq = self.__max_bigram()
        self.__clean_unigrams(max_bi_freq)
        for i in range(ngram_range[0], ngram_range[1]):
            self.__unbias_ngrams(i + 1)

        self.__lost_state = False
        self.__ngram_range = ngram_range

    @property
    def tfidf_matrix(self):
        return self.__tfidf_matrix

    @property
    def vectorizer(self):
        return self.__vectorizer

    @property
    def text(self):
        return self.__dataframe[self.__text_header]

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def dates(self):
        return list(self.__dataframe[self.__date_header])

    @property
    def doc_ids(self):
        return list(self.__dataframe[self.__id_header])