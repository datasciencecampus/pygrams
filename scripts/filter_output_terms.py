class FilterTerms(object):
    def __init__(self, term_score_tuples, nterms):
        print("filter terms")
        self.__nterms = nterms
        self.__term_score_tuples = term_score_tuples

    #embeddings best to be blended in the mask! ie. data[i] *= cosine_distance between domain words and ngram_average
    @property
    def term_score_tuples(self):
        return self.__term_score_tuples[:self.__nterms]