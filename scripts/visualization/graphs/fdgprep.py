import json
import os
from collections import Counter

from nltk import FreqDist
from nltk import bigrams


# A class able to prepare a graph, which then can be used in D3.js libraries to create a force-fdirected-graph (FDG)
# This code has been based on python code contributed by HMRC
class FDGPrep(object):

    def __init__(self, numwords):

        self.__numwords = numwords
        self.__abstracts_list = None
        self.__term_freq = None

    @staticmethod
    def __corpus_from_abstracts(raw_abstracts, common_words=None):

        corpus = []
        for patent_abstract in raw_abstracts:
            pattent_terms = []
            for term in patent_abstract:
                if term != '' and len(term) > 2:
                    if common_words is None or term in common_words:
                        pattent_terms.append(term)
            corpus.append(pattent_terms)
        return corpus

    def __create_graph_json(self):

        raw_comments = self.__abstracts_list

        corpus = raw_comments
        term_list = []
        for abstract in corpus:
            for term in abstract:
                term_list.append(term)

        term_freq = FreqDist(term_list)
        freq = term_freq.most_common(self.__numwords)

        common_words = [x[0] for x in freq]

        common_corpus = self.__corpus_from_abstracts(corpus, common_words)

        bigram_list = []
        for abstract in common_corpus:
            bigrams_list = bigrams(abstract)
            for bigram in bigrams_list:
                bigram_list.append(bigram)

        bigram_freq = zip(Counter(bigram_list).keys(), Counter(bigram_list).values())
        bigram_freq_list = list(bigram_freq)

        max_freq = 0.0
        max_bi_freq = 0.0

        min_freq = 10000.0
        min_bi_freq = 10000.0

        for word in self.__term_freq:
            if word[0] > max_freq:
                max_freq = word[0]
            if word[0] < min_freq:
                min_freq = word[0]

        for bigram in bigram_freq_list:
            if bigram[1] > max_bi_freq:
                max_bi_freq = bigram[1]
            if bigram[1] < min_bi_freq:
                min_bi_freq = bigram[1]

        node_results = []
        for word in self.__term_freq:
            if word[1] in common_words:
                score = (float(word[0]) - min_freq) / (float(max_freq) - min_freq)
                word_json = {'text': word[1], 'freq': score + 0.2}
                node_results.append(word_json)

        link_results = []
        for bigram in bigram_freq_list:
            bi = bigram[0]
            score = (float(bigram[1]) - min_bi_freq) / (float(max_bi_freq) - min_bi_freq)
            word_json = {'source': bi[0], 'target': bi[1], 'size': score + 0.2}
            link_results.append(word_json)

        graph = {'nodes': node_results, 'links': link_results}

        return graph

    def save_graph(self, fname, varname):

        graph = self.__create_graph_json()
        file_name = os.path.join('outputs', 'visuals', fname + '.js')
        with open(file_name, 'w') as js_temp:
            js_temp.write(varname + " = '[")
            json.dump(graph, js_temp)
            js_temp.write("]'")
        file_name_jason = os.path.join('outputs', 'reports', fname + '.json')
        with open(file_name_jason, 'w') as js_temp:
            json.dump(graph, js_temp)


    def fdg_tfidf(self, tf_idf, tf_idf2, args):
        num_terms_to_evaluate = 20
        abstracts = tf_idf.patent_abstracts
        sum_vector = tf_idf.get_tfidf_sum_vector()
        names = tf_idf.feature_names

        self.__abstracts_list = []
        single_terms = []
        for idx, abstract in enumerate(abstracts):
            terms, _, _ = tf_idf.detect_popular_ngrams_in_docs_set(number_of_ngrams_to_return=num_terms_to_evaluate,docs_set=[idx])
            if args.focus:
                terms2, _, _ = tf_idf2.extract_popular_ngrams(number_of_ngrams_to_return=num_terms_to_evaluate,
                                                              input_text=abstract)
                tset = set(terms)
                t2set = set(terms2)
                terms = tset - (tset.intersection(t2set))
            self.__abstracts_list.append(terms)
            for term in terms:
                single_terms.append(term)

        termsset = set(single_terms)

        self.__term_freq = []
        for index in range(len(names)):
            value = sum_vector[index]
            name = names[index]
            if name in termsset:
                feature_score_tuple = (value, names[index])
                self.__term_freq.append(feature_score_tuple)
