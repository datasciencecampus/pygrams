from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

import scripts.utils.utils as ut


class FilterTerms(object):
    def __init__(self, tfidf_obj, user_ngrams, file_name=None, threshold=None):

        self.__user_ngrams = user_ngrams
        self.__tfidf_ngrams = tfidf_obj.feature_names
        self.__file_name = file_name
        self.__ngram_weights_vec = list(np.ones(len(self.__tfidf_ngrams)))
        self.__tf_normalized = tfidf_obj.tf_matrix
        if self.__tf_normalized is not None:
            print('do stuff')

        if file_name is not None and user_ngrams is not None:
            print('Loading model: '+ file_name)
            self.__model = KeyedVectors.load_word2vec_format(self.__file_name)
            self.__ngram_weights_vec = self.__get_embeddings_vec(threshold)


    @property
    def ngram_weights_vec(self):
        return self.__ngram_weights_vec

    def __get_embeddings_vec(self, threshold):
        embeddings_vect = []
        user_terms = self.__user_ngrams.split(',')
        for term in tqdm(self.__tfidf_ngrams, desc='Evaluating terms distance with: ' + self.__user_ngrams, unit='term',
                         total=len(self.__tfidf_ngrams)):
            compare = []
            for ind_term in term.split():
                for user_term in user_terms:
                    try:
                        similarity_score = self.__model.similarity(ind_term, user_term)
                        compare.append(similarity_score)
                    except:
                        compare.append(0.0)
                        continue

            max_similarity_score = max(similarity_score for similarity_score in compare)
            embeddings_vect.append(max_similarity_score)
        embeddings_vect_norm = ut.normalize_array(embeddings_vect, return_list=True)
        if threshold is not None:
            return [float(int(x>threshold)) for x in embeddings_vect_norm]
        return embeddings_vect

