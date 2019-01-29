import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

class TermFocus():

    def __init__(self, tf_idf_in, tf_idf_random_in):
        self.__tfidf = tf_idf_in
        self.__tfidf_random = tf_idf_random_in

    def detect_and_focus_popular_ngrams(self,args, citation_count_dict, ngram_multiplier, num_ngrams, docs_set=None):
        pick = args.pick
        time = args.time
        focus = args.focus

        terms, ngrams_scores_tuple = self.__tfidf.detect_popular_ngrams_in_docs_set(
            number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
            pick=pick, time=time,
            citation_count_dict=citation_count_dict, docs_set=docs_set)

        if focus is None:
            print('No focus applied')
            focus_set_terms = set(terms)

        elif focus == 'set':
            print('Applying set difference focus')
            focus_set_terms = self.popular_ngrams_by_set_difference(number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
                                                               pick=pick, time=time,
                                                               citation_count_dict=citation_count_dict)
        elif focus == 'chi2':
            print('Applying chi2 focus')
            focus_terms, focus_scores = self.popular_ngrams_by_chi2_importance(ngram_multiplier * num_ngrams)
            focus_set_terms = set(focus_terms)

        elif focus == 'mutual':
            print('Applying mutual information focus')
            focus_terms, focus_scores = self.popular_ngrams_by_mutual_information(ngram_multiplier * num_ngrams)
            focus_set_terms = set(focus_terms)

        else:
            raise ValueError(f'Unknown focus type: {focus}')

        dict_freqs = dict([(p[0], (p[1])) for p in ngrams_scores_tuple if p[1] in focus_set_terms])

        return dict_freqs, focus_set_terms, ngrams_scores_tuple

    def popular_ngrams_by_set_difference(self, number_of_ngrams_to_return=200, pick='sum', time=False,
                                         citation_count_dict=None, docs_set=None):
        terms, _ = self.__tfidf.detect_popular_ngrams_in_docs_set(
            number_of_ngrams_to_return=number_of_ngrams_to_return,
            pick=pick, time=time, citation_count_dict=citation_count_dict, docs_set=docs_set)
        set_terms = set(terms)

        terms_random, _, = self.__tfidf_random.detect_popular_ngrams_in_docs_set(
            number_of_ngrams_to_return=number_of_ngrams_to_return,
            pick=pick, time=time, citation_count_dict=citation_count_dict, docs_set=docs_set)

        set_random_terms = set(terms_random)
        set_intersection = set_terms.intersection(set_random_terms)
        set_terms -= set_intersection

        return set_terms

    def popular_ngrams_by_chi2_importance(self, num_ngrams_report):
        return self.__popular_ngrams_with_selectkbest( num_ngrams_report, chi2)

    def popular_ngrams_by_mutual_information(self,  num_ngrams_report):
        return self.__popular_ngrams_with_selectkbest(num_ngrams_report, mutual_info_classif)


    def __popular_ngrams_with_selectkbest(self, num_ngrams_report, score_func):
        df = pd.DataFrame(self.__tfidf.abstracts, columns=['abstract'])
        df2 = pd.DataFrame(self.__tfidf_random.abstracts, columns=['abstract'])
        df['class_flag'] = 'Yes'
        df2['class_flag'] = 'No'
        df = pd.concat([df, df2])  # , sort=True)
        X = df['abstract'].values.astype('U')
        y = df['class_flag'].values.astype('U')

        ch2 = SelectKBest(score_func, k=num_ngrams_report)

        counter = self.__tfidf.vectorizer.fit_transform(X)
        tfidf_transformer = TfidfTransformer(smooth_idf=False)
        tfidf_matrix = tfidf_transformer.fit_transform(counter)
        ch2.fit(tfidf_matrix, y)

        ngram_chi2_tuples = [(self.__tfidf.vectorizer.get_feature_names()[i], ch2.scores_[i])
                             for i in ch2.get_support(indices=True)]
        ngram_chi2_tuples.sort(key=lambda tup: -tup[1])

        return [ngram_chi2_tuple[0]
                for ngram_chi2_tuple in ngram_chi2_tuples[:num_ngrams_report]
                if ngram_chi2_tuple[1] > 0], ngram_chi2_tuples[:num_ngrams_report]
