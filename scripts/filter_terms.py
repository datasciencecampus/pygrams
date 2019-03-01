from gensim.models import KeyedVectors
import numpy as np

class FilterTerms(object):
    def __init__(self, tfidf_ngrams, user_ngrams, file_name):

        self.__user_ngrams = user_ngrams
        self.__tfidf_ngrams = tfidf_ngrams
        self.__file_name = file_name
        if file_name is not None:
            self.model = KeyedVectors.load_word2vec_format(self.__file_name)
        self.__ngram_weights_vec = list(np.ones(len(tfidf_ngrams)))

    @property
    def ngram_weights_vec(self):
        return self.__ngram_weights_vec

    # no need for both threshold and binary as threshold only needed in binary :)
    def get_embeddings_vec(self, threshold=1.5):
        embeddings_vect = []
        for term in self.__tfidf_ngrams:
            compare = []
            for ind_term in term.split():
                # what if the user put an ngram there?
                # Maybe address in the future, keep it simple for now
                for user_term in self.__user_ngrams.split(','):
                    try:
                        j = self.model.similarity(ind_term, user_term)
                        compare.append(j)
                    except:
                        continue
            append_val = 1.0
            if len(compare) > 0:
                x = max(float(s) for s in compare)
                append_val = x if threshold is None else int(x > threshold)
            embeddings_vect.append(append_val)
        return embeddings_vect

