class FilterTerms(object):
    def __init__(self, tfidf_ngrams, user_ngrams, binary=False, threshold = None):
        print("filter terms")
        self.__user_ngrams = user_ngrams
        self.__tfidf_ngrams = tfidf_ngrams
        self.__ngrams_weights_vect = None

    #embeddings best to be blended in the mask! ie. data[i] *= cosine_distance between domain words and ngram_average
    @property
    def ngrams_weights_vect(self):
        return self.__ngrams_weights_vect

    def calculate_distances(self):
        print()
        #populate this one: self.__ngrams_weights_vect