import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif


def detect_and_focus_popular_ngrams(pick, time, focus, citation_count_dict, ngram_multiplier, num_ngrams, tfidf, tfidf_random):
    terms, ngrams_scores_tuple = tfidf.detect_popular_ngrams_in_corpus(
        number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
        pick=pick, time=time,
        citation_count_dict=citation_count_dict)

    if focus is None:
        print('No focus applied')
        focus_set_terms = set(terms)

    elif focus == 'set':
        print('Applying set difference focus')
        focus_set_terms = popular_ngrams_by_set_difference(tfidf, tfidf_random,
                                                           number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
                                                           pick=pick, time=time,
                                                           citation_count_dict=citation_count_dict)
    elif focus == 'chi2':
        print('Applying chi2 focus')
        focus_terms, focus_scores = popular_ngrams_by_chi2_importance(tfidf, tfidf_random,
                                                                            ngram_multiplier * num_ngrams)
        focus_set_terms = set(focus_terms)

    elif focus == 'mutual':
        print('Applying mutual information focus')
        focus_terms, focus_scores = popular_ngrams_by_mutual_information(tfidf, tfidf_random,
                                                                               ngram_multiplier * num_ngrams)
        focus_set_terms = set(focus_terms)

    else:
        raise ValueError(f'Unknown focus type: {args.focus}')

    dict_freqs = dict([((p[1]), p[0]) for p in ngrams_scores_tuple if p[1] in focus_set_terms])

    return dict_freqs, focus_set_terms, ngrams_scores_tuple


def popular_ngrams_by_set_difference(tfidf, tfidf_random,
                                     number_of_ngrams_to_return=200, pick='sum', time=False,
                                     citation_count_dict=None):
    terms, ngrams_scores_tuple = tfidf.detect_popular_ngrams_in_corpus(
        number_of_ngrams_to_return=number_of_ngrams_to_return,
        pick=pick, time=time, citation_count_dict=citation_count_dict)
    set_terms = set(terms)

    terms_random, ngrams_scores_tuple_random = tfidf_random.detect_popular_ngrams_in_corpus(
        number_of_ngrams_to_return=number_of_ngrams_to_return,
        pick=pick, time=time, citation_count_dict=citation_count_dict)

    set_random_terms = set(terms_random)
    set_intersection = set_terms.intersection(set_random_terms)
    set_terms -= set_intersection

    return set_terms


def popular_ngrams_by_chi2_importance(tfidf, tfidf_random, num_ngrams_report):
    return __popular_ngrams_with_selectkbest(tfidf, tfidf_random, num_ngrams_report, chi2)


def popular_ngrams_by_mutual_information(tfidf, tfidf_random, num_ngrams_report):
    return __popular_ngrams_with_selectkbest(tfidf, tfidf_random, num_ngrams_report, mutual_info_classif)


def __popular_ngrams_with_selectkbest(tfidf, tfidf_random, num_ngrams_report, score_func):
    df = pd.DataFrame(tfidf.patent_abstracts, columns=['abstract'])
    df2 = pd.DataFrame(tfidf_random.patent_abstracts, columns=['abstract'])
    df['class_flag'] = 'Yes'
    df2['class_flag'] = 'No'
    df = pd.concat([df, df2], sort=True)
    X = df['abstract'].values.astype('U')
    y = df['class_flag'].values.astype('U')
    tfidf_vectorizer = tfidf.tfidf_vectorizer

    ch2 = SelectKBest(score_func, k=num_ngrams_report)

    tfidf_matrix = tfidf_vectorizer.fit_transform(X)
    ch2.fit(tfidf_matrix, y)
    # pipe = make_pipeline_imb(tfidf_vectorizer, ch2)
    # pipe.fit(X, y)
    feature_names = [tfidf_vectorizer.get_feature_names()[i] for i in ch2.get_support(indices=True)]
    feature_chi_scores = [ch2.scores_[i] for i in ch2.get_support(indices=True)]

    ngram_chi2_tuples = list(zip(feature_names, feature_chi_scores))

    ngram_chi2_tuples.sort(key=lambda tup: -tup[1])

    return [ngram_chi2_tuple[0]
            for ngram_chi2_tuple in ngram_chi2_tuples[:num_ngrams_report]
            if ngram_chi2_tuple[1] > 0], ngram_chi2_tuples[:num_ngrams_report]


def old_code(ch2, df, feature_names, feature_chi_scores, tfidf_matrix, num_ngrams_report, freqs, terms1,
             tfidf_vectorizer):
    # feature_pvalues = [ch2.pvalues_[i] for i in ch2.get_support(indices=True)]
    yes_score = []
    no_score = []
    for i in ch2.get_support(indices=True):
        df['feature_scores'] = tfidf_matrix[:, i].toarray()
        # print()
        # print(i, tfidf_matrix[:, i].toarray())
        yes_score.append(df[(df.class_flag == 'Yes') & (df.feature_scores != 0.0)]['feature_scores'].sum())
        no_score.append(df[(df.class_flag == 'No') & (df.feature_scores != 0.0)]['feature_scores'].sum())
    df3 = pd.DataFrame({'feature_names': feature_names,
                        'feature_scores': feature_chi_scores,
                        # 'feature_pvalues': feature_pvalues,
                        'yes_score': yes_score,
                        'no_score': no_score,
                        }).sort_values(by='feature_scores', ascending=False)
    print()
    print('chi-2')
    print()
    for i in range(num_ngrams_report):
        print('{:30} {:f} {:f} {:f} {:f}'.format(df3.feature_names.iloc[i],
                                                 df3.feature_scores.iloc[i],
                                                 # df3.feature_pvalues.iloc[i],
                                                 df3.yes_score.iloc[i],
                                                 df3.no_score.iloc[i],
                                                 df3.yes_score.iloc[i] / df3.no_score.iloc[i]
                                                 ))
    print()
    common = 0
    common_features = []
    new_features = []
    for i in range(num_ngrams_report):
        if not freqs[i][1] in feature_names:
            print('    ', end="")
            common += 1
            common_features.append(freqs[i][1])
        print('{:30} {:f}'.format(freqs[i][1], freqs[i][0]))
        if not feature_names[i] in terms1:
            new_features.append(feature_names[i])
    print()
    print('non-common features =', common)
    print()
    my_set = set(common_features)
    print(*common_features, sep="\n")  # this doesn't sort the list
    # print("\n".join(common_features))  # this does sort the list
    print()
    print('new features =', len(new_features))
    print()
    print(*new_features, sep="\n")  # this doesn't sort the list

    # Plots of non-zero tfidf values
    print(df.shape)
    print(df[df.class_flag == 'Yes'].shape)
    print(df[df.class_flag == 'No'].shape)
    feature_names = tfidf_vectorizer.get_feature_names()
    my_feature = 'hybrid vehicle'  # first / battery, control unit/fuel cell, control device/solar cell
    my_index = feature_names.index(my_feature)
    my_X = tfidf_matrix[:, my_index].toarray()
    df4 = df.copy()
    df4['tfs'] = my_X
    df4 = df4[df4.tfs > 0.0]
    print(df4.shape, 'my_feature =', my_feature,
          'Y02 No =', df4[df4.class_flag == 'No'].shape[0], df4[df4.class_flag == 'No'].tfs.mean(),
          'Y02 Yes =', df4[df4.class_flag == 'Yes'].shape[0], df4[df4.class_flag == 'Yes'].tfs.mean())
    df4.groupby('class_flag').tfs.hist(alpha=0.4)
    # df4.groupby('class_flag').hist(column='tfs', alpha=0.4)
    df4.boxplot(column='tfs', by='class_flag')
    # df4.groupby('class_flag').boxplot(column='tfs')
    # plt.title("term = '" + my_feature + "'")
    # plt.xlabel('tfidf', size=13)
    # plt.ylabel('frequency', size=13)
    # # plt.legend(['Random', 'Y02'])
    # plt.show()

    # return common_features
