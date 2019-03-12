from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scripts.text_processing import StemTokenizer, lowercase_strip_accents_and_ownership, WordAnalyzer


class TFIDF:

    def __init__(self, docs_df, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer(),
                 text_header='abstract'):

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

        self.__text_header = text_header

        num_docs_before_sift = self.__dataframe.shape[0]
        self.__dataframe.dropna(subset=[self.__text_header], inplace=True)
        num_docs_after_sift = self.__dataframe.shape[0]
        num_docs_sifted = num_docs_before_sift - num_docs_after_sift
        print(f'Dropped {num_docs_sifted:,} from {num_docs_before_sift:,} docs due to empty text field')

        self.__ngram_counts = self.__vectorizer.fit_transform(self.__dataframe[self.__text_header])
        self.__feature_names = self.__vectorizer.get_feature_names()

        self.__tfidf_transformer = TfidfTransformer(smooth_idf=False)
        self.__tfidf_matrix = self.__tfidf_transformer.fit_transform(self.__ngram_counts)

    @property
    def dataframe(self):
        return self.__dataframe

    @property
    def idf(self):
        return self.__tfidf_transformer.idf_

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
