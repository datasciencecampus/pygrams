from scripts.text_processing import StemTokenizer


class Dice(object):

    def __init__(self, expected, tokenizer=StemTokenizer()):
        self.__tokenizer = tokenizer

        self.__expected_token_unigrams = []
        self.__expected_token_ngrams = []
        self.__expected_token_bigrams = []
        for expected_term in expected:
            tokens = self.__tokenizer(expected_term.lower())
            ngram = ' '.join(tokens)
            for token in tokens:
                self.__expected_token_unigrams.append(token)
            self.__expected_token_ngrams.append(ngram)
            if len(tokens) == 2:
                self.__expected_token_bigrams.append(ngram)
        self.__expected_token_unigrams = list(set(self.__expected_token_unigrams))

    @property
    def expected_token_ngrams(self):
        return self.__expected_token_ngrams

    @property
    def expected_token_unigrams(self):
        return self.__expected_token_unigrams

    @property
    def expected_token_bigrams(self):
        return self.__expected_token_bigrams

    def score(self, expectedset, actualset):
        TP = expectedset.intersection(actualset)
        FN = expectedset.difference(actualset)
        FP = actualset.difference(expectedset)
        return 2 * len(TP) / (2 * len(TP) + len(FP) + len(FN)), actualset, TP, FN, FP

    def get_score_ngrams(self, actual):
        actual_ngrams_set = set(actual)
        expected_ngrams_set = set(self.__expected_token_ngrams)
        return self.score(expected_ngrams_set, actual_ngrams_set)

    def get_score_unigrams(self, actual):

        actual_unigrams = []
        for actual_term in actual:
            tokens = self.__tokenizer(actual_term.lower())
            for token in tokens:
                actual_unigrams.append(token)

        actual_unigrams_set = set(actual_unigrams)
        expected_unigrams_set = set(self.__expected_token_unigrams)

        return self.score(expected_unigrams_set, actual_unigrams_set)

    def get_score_bigrams(self, actual):

        actual_bigrams = []
        for actual_term in actual:
            tokens = self.__tokenizer(actual_term.lower())
            ngram = ' '.join(tokens)
            if len(tokens) == 2:
                actual_bigrams.append(ngram)

        actual_bigrams_set = set(actual_bigrams)
        expected_bigrams_set = set(self.__expected_token_bigrams)

        return self.score(expected_bigrams_set, actual_bigrams_set)
