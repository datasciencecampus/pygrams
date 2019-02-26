from gensim.models import KeyedVectors

class FilterTerms(object):
    def __init__(self, tfidf_ngrams, user_ngrams, file_path, file_name, model=None, binary_vect = list(),
                 distance_vect = list(), binary=False, threshold = None):
        print("filter terms")
        self.__user_ngrams = user_ngrams
        print('user ngrams = ', self.__user_ngrams)
        self.__tfidf_ngrams = tfidf_ngrams
        self.__file_path = file_path
        self.__file_name = file_name
        self.binary_vect = binary_vect
        self.distance_vect = distance_vect
        self.model = model
        self.binary = binary
        self.threshold = threshold

        print('tfidf ngrams = ', self.__tfidf_ngrams)
        print('length of tfidf_ngrams = ',len(self.__tfidf_ngrams))
        self.load_fasttext_model()

        if not binary:
            for term in self.__tfidf_ngrams:
                print('term = ', term)
                compare = []
                for ind_term in term.split():
                    print('ind_term = ', ind_term)
                    for user_term in self.__user_ngrams.split(','):
                        print('user_term =', user_term)
                        try:
                            j = self.calculate_distances(ind_term, user_term)
                            print('j=', j)
                            compare.append(j)
                        except:
                            continue
                print('compare =', compare)
                if compare:
                    x = max(float(s) for s in compare)
                    print(x)
                    if x > self.threshold:
                        self.distance_vect.append(x)
                    else:
                        self.distance_vect.append(0)
                else:
                    self.distance_vect.append(0)
        print(self.distance_vect)
        print('length of distanct vect =', len(self.distance_vect))
        print('length of tfidf_ngrams = ', len(self.__tfidf_ngrams))

        if binary:
            for term in self.__tfidf_ngrams:
                print('term = ', term)
                compare = []
                for ind_term in term.split():
                    print('ind_term = ', ind_term)
                    for user_term in self.__user_ngrams.split(','):
                        print('user_term =', user_term)
                        try:
                            j = self.calculate_distances(ind_term, user_term)
                            print('j=', j)
                            compare.append(j)
                        except:
                            continue
                print('compare =', compare)
                if compare:
                    x = max(float(s) for s in compare)
                    print(x)
                    if x > self.threshold:
                        self.binary_vect.append(1)
                    else:
                        self.binary_vect.append(0)
                else:
                    self.binary_vect.append(0)
            print(self.binary_vect)
            print('length of binary vect =', len(self.binary_vect))
            print('length of tfidf_ngrams = ', len(self.__tfidf_ngrams))

    def load_fasttext_model(self):
        self.model = KeyedVectors.load_word2vec_format(self.__file_name)

    def calculate_distances(self, i, p):
        return self.model.similarity(i, p)
        # populate this one: self.__ngrams_weights_vect