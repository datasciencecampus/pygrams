import copy
import datetime
import string

import numpy as np
from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, strip_accents_ascii
from tqdm import tqdm
from nltk.corpus import wordnet
from scripts import FilePaths
from scripts.utils.utils import Utils as ut

"""Sections of this code are based on scikit-learn sources; scikit-learn code is covered by the following license:
New BSD License

Copyright (c) 2007–2018 The scikit-learn developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Scikit-learn Developers  nor the names of
     its contributors may be used to endorse or promote products
     derived from this software without specific prior written
     permission. 


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def lemmatize_with_pos(self, tag):
        if tag[1].startswith('N'):
            return self.wnl.lemmatize(tag[0], wordnet.NOUN)
        elif tag[1].startswith('J'):
            return self.wnl.lemmatize(tag[0], wordnet.ADJ)
        elif tag[1].startswith('R'):
            return self.wnl.lemmatize(tag[0], wordnet.ADV)
        elif tag[1].startswith('V'):
            return self.wnl.lemmatize(tag[0], wordnet.VERB)
        else:
            return self.wnl.lemmatize(tag[0])

    def __call__(self, doc):
        text = word_tokenize(doc)
        pos_tagged_tokens = pos_tag(text)
        return [self.lemmatize_with_pos(t) for t in pos_tagged_tokens]


class StemTokenizer(object):
    def __init__(self):
        self.ps = PorterStemmer()

    def __call__(self, doc):
        return [self.ps.stem(t) for t in word_tokenize(doc)]


def lowercase_strip_accents_and_ownership(doc):
    lowercase_no_accents_doc = strip_accents_ascii(doc.lower())
    return lowercase_no_accents_doc.replace("'s", "")


class WordAnalyzer(object):
    tokenizer = None
    preprocess = None
    ngram_range = None

    @staticmethod
    def init(tokenizer, preprocess, ngram_range):
        WordAnalyzer.tokenizer = tokenizer
        WordAnalyzer.preprocess = preprocess
        WordAnalyzer.ngram_range = ngram_range

        # global stopwords
        with open(FilePaths.global_stopwords_filename, 'r') as f:
            stemmed_stop_word_set_glob = set(WordAnalyzer.tokenizer(f.read())).union(list(string.punctuation))

        # we want to stop some words on n>1 ngrams
        with open(FilePaths.ngram_stopwords_filename, 'r') as f:
            WordAnalyzer.stemmed_stop_word_set_n = set(WordAnalyzer.tokenizer(f.read())).union(
                list(stemmed_stop_word_set_glob))

        with open(FilePaths.unigram_stopwords_filename, 'r') as f:
            WordAnalyzer.stemmed_stop_word_set_uni = set(WordAnalyzer.tokenizer(f.read())).union(
                list(stemmed_stop_word_set_glob))

    # Based on VectorizeMixin in sklearn text.py
    @staticmethod
    def analyzer(doc):
        """based on VectorizerMixin._word_ngrams in sklearn/feature_extraction/text.py,
        from scikit-learn; extended to prevent generation of n-grams containing stop words"""
        tokens = WordAnalyzer.tokenizer(WordAnalyzer.preprocess(doc))

        # handle token n-grams
        min_n, max_n = WordAnalyzer.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = [w for w in tokens if w not in WordAnalyzer.stemmed_stop_word_set_uni and not w.isdigit()]
                # tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    candidate_ngram = original_tokens[i: i + n]
                    hasdigit = False
                    for ngram in candidate_ngram:
                        if ngram.isdigit():
                            hasdigit = True

                    ngram_stop_word_set = set(candidate_ngram) & WordAnalyzer.stemmed_stop_word_set_n
                    if len(ngram_stop_word_set) == 0 and not hasdigit:
                        tokens_append(space_join(candidate_ngram))

            return tokens
        else:
            return [w for w in tokens if w not in WordAnalyzer.stemmed_stop_word_set_uni]


class TFIDF:
    def __init__(self, patentsdf, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer(), header='abstract'):
        self.__dataframe = patentsdf

        WordAnalyzer.init(
            tokenizer=tokenizer,
            preprocess=lowercase_strip_accents_and_ownership,
            ngram_range=ngram_range)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=max_document_frequency,
            min_df=1,
            ngram_range=ngram_range,
            analyzer=WordAnalyzer.analyzer
        )
        
        self.__abstract_header=header

        number_of_patents_before_sift = self.__dataframe.shape[0]
        self.__dataframe.dropna(subset=[self.__abstract_header], inplace=True)
        number_of_patents_after_sift = self.__dataframe.shape[0]
        number_of_patents_sifted = number_of_patents_before_sift - number_of_patents_after_sift
        print(f'Dropped {number_of_patents_sifted:,} patents due to empty abstracts')

        self.tfidf_vectorizer.fit(self.__dataframe[self.__abstract_header])
        self.__feature_names = self.tfidf_vectorizer.get_feature_names()

        self.tfidf_matrix = self.tfidf_vectorizer.transform(self.__dataframe[self.__abstract_header])

        self.tfidf_matrix = self.unbias_ngrams(self.tfidf_matrix, ngram_range[0])
        self.__lost_state = False
        self.__min_ngram_count = ngram_range[0]

    @property
    def tfidf_mat(self):
        return self.tfidf_matrix

    @property
    def patent_abstracts(self):
        return self.__dataframe[self.__abstract_header]

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def publication_dates(self):
        return list(self.__dataframe['publication_date'])

    @property
    def patent_ids(self):
        return list(self.__dataframe['patent_id'])

    def detect_popular_ngrams_in_docs_set(self, number_of_ngrams_to_return=200, pick='sum', time=False,
                                          citation_count_dict=None, docs_set=None):
        num_docs = 0 if docs_set is None else len(docs_set)
        print(f'Processing TFIDF of {num_docs} / {self.tfidf_matrix.shape[0]:,} documents')

        if self.tfidf_matrix.shape[0] == 0:
            print('...skipping as 0 patents...')
            return []

        if self.__lost_state:

            self.tfidf_vectorizer.fit(self.__dataframe[self.__abstract_header])
            self.__feature_names = self.tfidf_vectorizer.get_feature_names()

            self.tfidf_matrix = self.tfidf_vectorizer.transform(self.__dataframe[self.__abstract_header])

            self.tfidf_matrix = self.unbias_ngrams(self.tfidf_matrix, self.__min_ngram_count)
            self.__lost_state = False


        if time:
            self.__dataframe = self.__dataframe.sort_values(by=['publication_date'])
            epoch = datetime.datetime.utcfromtimestamp(0)
            num_docs = len(self.__dataframe['publication_date'])
            time_weights = [0.0] * num_docs
            for patent_index, pub_date in enumerate(self.__dataframe['publication_date']):
                time_weights[patent_index] = (pub_date - epoch).total_seconds()

            mx = max(time_weights)
            mn = min(time_weights)

            for patent_index, pub_date in enumerate(self.__dataframe['publication_date']):
                X = time_weights[patent_index]
                X_std = (X - mn) / (mx - mn)
                time_weights[patent_index] = X_std

            for i, v in enumerate(time_weights):
                self.tfidf_matrix.data[self.tfidf_matrix.indptr[i]:self.tfidf_matrix.indptr[i + 1]] *= v
                self.__lost_state=True

        if citation_count_dict:
            #TODO check if we need -2 below. If not, we only need one dict for both citations and docs_set
            patent_id_dict = {k[:-2]: v for v, k in enumerate(self.__dataframe.patent_id)}
            citation_count_for_patent_id_dict = {}
            for key, _ in tqdm(patent_id_dict.items()):
                citation_count_for_patent_id_dict[key] = citation_count_dict.get(key, .0)

            max_citation_count_val = float(max(citation_count_for_patent_id_dict.values()))
            min_citation_count_val = 0.05

            if max_citation_count_val == 0:
                for patent_id in citation_count_for_patent_id_dict:
                    citation_count_for_patent_id_dict[patent_id] = 1.0
            else:
                for patent_id in citation_count_for_patent_id_dict:
                    citation_count_for_patent_id_dict_std = min_citation_count_val + (
                            (float(citation_count_for_patent_id_dict[patent_id]) - min_citation_count_val) / (
                            max_citation_count_val - min_citation_count_val))
                    citation_count_for_patent_id_dict[patent_id] = citation_count_for_patent_id_dict_std

            list_of_citation_counts = list(citation_count_for_patent_id_dict.values())

            for i, v in enumerate(list_of_citation_counts):
                self.tfidf_matrix.data[self.tfidf_matrix.indptr[i]:self.tfidf_matrix.indptr[i + 1]] *= v
            self.__lost_state = True


        # pick filter
        tfidf_csc_matrix = self.tfidf_matrix.tocsc()

        if pick == 'median':
            pick_func = np.median
        elif pick == 'avg':
            pick_func = np.average
        elif pick == 'max':
            pick_func = np.max
        elif pick == 'sum':
            pick_func = np.sum

        ngrams_scores_tuple = []
        for ngram_index, ngram in enumerate(
                tqdm(self.__feature_names, leave=False, desc='Searching TFIDF', unit='ngram')):

            non_zero_values_term = tfidf_csc_matrix.data[
                                  tfidf_csc_matrix.indptr[ngram_index]:tfidf_csc_matrix.indptr[ngram_index + 1]
                                  ]
            if docs_set is not None:

                row_indices_term = tfidf_csc_matrix.indices[tfidf_csc_matrix.indptr[ngram_index]:tfidf_csc_matrix.indptr[ngram_index + 1]]
                non_zero_values_term_set=[]

                indices_idx=0
                for doc_idx in docs_set:
                    while indices_idx <= doc_idx and indices_idx < len(row_indices_term):
                        if row_indices_term[indices_idx] == doc_idx:
                            non_zero_values_term_set.append(non_zero_values_term[indices_idx])
                        indices_idx += 1
                non_zero_values_term = non_zero_values_term_set

            if len(non_zero_values_term)>0:
                pick_value = pick_func(non_zero_values_term)

                if np.isnan(pick_value):
                    pick_value = 0

                ngrams_scores_tuple.append((pick_value, ngram))

        ngrams_scores_tuple.sort(key=lambda tup: -tup[0])

        return [feature_score_tuple[1]
                for feature_score_tuple in ngrams_scores_tuple[:number_of_ngrams_to_return]
                if feature_score_tuple[0] > 0], ngrams_scores_tuple[:number_of_ngrams_to_return], self.tfidf_matrix

    def get_tfidf_sum_vector(self):
        tfidf = self.tfidf_vectorizer.transform(self.patent_abstracts)
        tfidf_summary = (tfidf.sum(axis=0)).flatten()
        return tfidf_summary.tolist()[0]

    def unbias_ngrams(self, mtx_csr, min_ngram_len):

        # iterate through rows ( docs)
        for i in range(0, len(mtx_csr.indptr) -1):
            start_idx_ptr = mtx_csr.indptr[i]
            end_idx_ptr = mtx_csr.indptr[i + 1]

            # iterate through columns with non-zero entries
            for j in range(start_idx_ptr, end_idx_ptr):

                col_idx = mtx_csr.indices[j]
                big_ngram = self.feature_names[col_idx]
                big_ngram_terms = big_ngram.split()

                if len(big_ngram_terms) > min_ngram_len:

                    ngram_minus_front = ' '.join(big_ngram_terms[1:])
                    ngram_minus_back = ' '.join(big_ngram_terms[:len(big_ngram_terms) - 1])

                    j_1 = j+1
                    col_idx_1 = mtx_csr.indices[j_1]
                    prev_nonzero_ngram = self.feature_names[col_idx_1]

                    while prev_nonzero_ngram > ngram_minus_back and j_1 >= start_idx_ptr:
                        j_1 += 1
                        col_idx_1 = mtx_csr.indices[j_1]
                        prev_nonzero_ngram = self.feature_names[col_idx_1]

                    if prev_nonzero_ngram == ngram_minus_back:
                        mtx_csr.data[j_1] = 0.0

                    if big_ngram > ngram_minus_front:
                        start, end = 0, col_idx-1
                    else:
                        start, end = col_idx+1, len(self.feature_names)-1

                    term_idx, found = ut.bisearch_csr(self.feature_names, ngram_minus_front, start, end)

                    if found and term_idx in mtx_csr.indices[start_idx_ptr:end_idx_ptr]:
                        idx = mtx_csr.indices[start_idx_ptr:end_idx_ptr].tolist().index(term_idx)
                        mtx_csr.data[start_idx_ptr + idx] = 0.0

        return mtx_csr
