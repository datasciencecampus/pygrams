import os
import zipfile

import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

import scripts.utils.utils as ut


class FilterTerms(object):
    def __init__(self, tfidf_ngrams, user_ngrams, file_name=os.path.join('models', 'glove', 'glove2vec.6B.50d.txt'),
                 threshold=None):

        self.__user_ngrams = user_ngrams
        self.__tfidf_ngrams = tfidf_ngrams
        self.__file_name = file_name
        self.__ngram_weights_vec = list(np.ones(len(tfidf_ngrams)))
        if user_ngrams is not None and len(user_ngrams)>0:
            if not os.path.isfile(file_name):
                output_path = os.path.dirname(file_name)
                with zipfile.ZipFile(file_name + ".zip", "r") as zip_ref:
                    zip_ref.extractall(path=output_path)
            self.__model = KeyedVectors.load_word2vec_format(self.__file_name)
            self.__ngram_weights_vec = self.__get_embeddings_vec(threshold)

    @property
    def ngram_weights_vec(self):
        return self.__ngram_weights_vec

    def __get_embeddings_vec(self, threshold):
        embeddings_vect = []
        for term in tqdm(self.__tfidf_ngrams, desc='Evaluating terms distance with: ' + ' '.join(self.__user_ngrams), unit='term',
                         total=len(self.__tfidf_ngrams)):
            compare = []
            for ind_term in term.split():
                for user_term in self.__user_ngrams:
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
            return [float(x>threshold) for x in embeddings_vect_norm]
        return embeddings_vect

