import datetime
import string

import numpy as np
from nltk import word_tokenize, PorterStemmer, pos_tag
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import strip_accents_ascii, TfidfTransformer
from tqdm import tqdm
from nltk.corpus import wordnet
from scripts import FilePaths
from sklearn.feature_extraction.text import CountVectorizer


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
    txt= lowercase_no_accents_doc.replace('"', '').replace("\'s", "").replace("\'ve", " have").replace("\'re", " are").replace("\'", "").strip("`").strip()
    return txt


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
    def __init__(self, docs_df, ngram_range=(1, 3), max_document_frequency=0.3, tokenizer=StemTokenizer(), header='abstract', normalize_doc_length=False):
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

        self.__abstract_header = header

        num_docs_before_sift = self.__dataframe.shape[0]
        self.__dataframe.dropna(subset=[self.__abstract_header], inplace=True)
        num_docs_after_sift = self.__dataframe.shape[0]
        num_docs_sifted = num_docs_before_sift - num_docs_after_sift
        print(f'Dropped {num_docs_sifted:,} docs due to empty abstracts')

        self.__ngram_counts = self.__vectorizer.fit_transform(self.__dataframe[self.__abstract_header])
        self.__feature_names = self.__vectorizer.get_feature_names()

        if normalize_doc_length:
            self.__ngram_counts = csr_matrix(self.__ngram_counts, dtype=np.float64, copy=True)
            self.__normalize_rows()

        self.__tfidf_transformer = TfidfTransformer(smooth_idf=False)
        self.__tfidf_matrix = self.__tfidf_transformer.fit_transform(self.__ngram_counts)
        for i in range(ngram_range[0], ngram_range[1]):
            self.__unbias_ngrams(i+1)
        self.__lost_state = False
        self.__ngram_range = ngram_range


    @property
    def tfidf_matrix(self):
        return self.__tfidf_matrix

    @property
    def vectorizer(self):
        return self.__vectorizer

    @property
    def abstracts(self):
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
        if docs_set is None:
            print(f'Processing TFIDF of {self.__tfidf_matrix.shape[0]:,} documents')

        if self.__tfidf_matrix.shape[0] == 0:
            print('...skipping as 0 patents...')
            return []

        if self.__lost_state:
            self.__tfidf_matrix = self.__tfidf_transformer.fit_transform(self.__ngram_counts)
            for i in range(self.__ngram_range[0], self.__ngram_range[1]):
                 self.__unbias_ngrams(i + 1)
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
                self.__tfidf_matrix.data[self.__tfidf_matrix.indptr[i]:self.__tfidf_matrix.indptr[i + 1]] *= v
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
                self.__tfidf_matrix.data[self.__tfidf_matrix.indptr[i]:self.__tfidf_matrix.indptr[i + 1]] *= v
            self.__lost_state = True

        # pick filter
        tfidf_csc_matrix = self.__tfidf_matrix.tocsc()

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

            start_idx_inptr = tfidf_csc_matrix.indptr[ngram_index]
            end_idx_inptr = tfidf_csc_matrix.indptr[ngram_index+1]

            non_zero_values_term = tfidf_csc_matrix.data[start_idx_inptr:end_idx_inptr]

            if docs_set is not None:

                row_indices_term = tfidf_csc_matrix.indices[start_idx_inptr:end_idx_inptr]
                non_zero_values_term_set=[]

                indices_idx=0
                for doc_idx in docs_set:
                    while indices_idx < len(row_indices_term) and row_indices_term[indices_idx] <= doc_idx:
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

        ngrams_scores_slice = ngrams_scores_tuple[:number_of_ngrams_to_return]

        return [feature_score_tuple[1] for feature_score_tuple in ngrams_scores_slice
                if feature_score_tuple[0] > 0], ngrams_scores_slice

    def __normalize_rows(self):

        for idx, text in enumerate(self.abstracts):
            text_len = len(text)
            self.__ngram_counts.data[self.__ngram_counts.indptr[idx]: self.__ngram_counts.indptr[idx + 1]] /= text_len

    def __unbias_ngrams(self, ngram_length):

        # iterate through rows ( docs)
        for i in range(len(self.abstracts)):
            start_idx_ptr = self.__tfidf_matrix.indptr[i]
            end_idx_ptr = self.__tfidf_matrix.indptr[i + 1]

            # iterate through columns with non-zero entries
            for j in range(start_idx_ptr, end_idx_ptr):

                col_idx = self.__tfidf_matrix.indices[j]
                big_ngram = self.__feature_names[col_idx]
                big_ngram_terms = big_ngram.split()

                if len(big_ngram_terms) == ngram_length:

                    ngram_minus_front = ' '.join(big_ngram_terms[1:])
                    ngram_minus_back  = ' '.join(big_ngram_terms[:len(big_ngram_terms) - 1])
                    idx_ngram_minus_front = self.__vectorizer.vocabulary_.get(ngram_minus_front)
                    idx_ngram_minus_back  = self.__vectorizer.vocabulary_.get(ngram_minus_back)

                    indices_slice = self.__tfidf_matrix.indices[start_idx_ptr:end_idx_ptr]
                    ngram_counts = self.__tfidf_matrix.data[j]

                    self.__unbias_ngrams_slice(indices_slice, idx_ngram_minus_front, ngram_counts, start_idx_ptr)
                    self.__unbias_ngrams_slice(indices_slice, idx_ngram_minus_back, ngram_counts, start_idx_ptr)

    def __unbias_ngrams_slice(self, dindices_slice, idx_ngram, ngram_counts, start_idx_ptr):
        if idx_ngram in dindices_slice:
            idx = dindices_slice.tolist().index(idx_ngram)

            if ngram_counts < self.__tfidf_matrix.data[start_idx_ptr + idx]:
                self.__tfidf_matrix.data[start_idx_ptr + idx] -= ngram_counts
            else:
                self.__tfidf_matrix.data[start_idx_ptr + idx] = 0
