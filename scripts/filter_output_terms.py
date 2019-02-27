from gensim.models import KeyedVectors

class FilterTerms(object):
    def __init__(self, tfidf_ngrams, user_ngrams, file_name, model=None, binary_vect = list(),
                 distance_vect = list(), binary=False, threshold = None):
        print("filter terms")
        self.__user_ngrams = user_ngrams
        self.__tfidf_ngrams = tfidf_ngrams
        self.__file_name = file_name
        self.binary_vect = binary_vect
        self.distance_vect = distance_vect
        self.model = model
        self.binary = binary
        self.threshold = threshold

        self.load_fasttext_model()

        if not binary:
            for term in self.__tfidf_ngrams:
                compare = []
                for ind_term in term.split():
                    for user_term in self.__user_ngrams.split(','):
                        try:
                            j = self.calculate_distances(ind_term, user_term)
                            compare.append(j)
                        except:
                            continue
                if compare:
                    x = max(float(s) for s in compare)
                    if x > self.threshold:
                        self.distance_vect.append(x)
                    else:
                        self.distance_vect.append(0)
                else:
                    self.distance_vect.append(0)

        if binary:
            for term in self.__tfidf_ngrams:
                compare = []
                for ind_term in term.split():
                    for user_term in self.__user_ngrams.split(','):
                        try:
                            j = self.calculate_distances(ind_term, user_term)
                            compare.append(j)
                        except:
                            continue
                if compare:
                    x = max(float(s) for s in compare)
                    print(x)
                    if x > self.threshold:
                        self.binary_vect.append(1)
                    else:
                        self.binary_vect.append(0)
                else:
                    self.binary_vect.append(0)

    def load_fasttext_model(self):
        self.model = KeyedVectors.load_word2vec_format(self.__file_name)

    def calculate_distances(self, i, p):
        return self.model.similarity(i, p)