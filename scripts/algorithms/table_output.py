import pandas as pd


def table_output(tfidf, tfidf_random, citation_count_dict, num_ngrams, pick, ngram_multiplier, table_name):
    """
    Description: Creates a table showing changes in feature rankings with focus, time and cite options
    Receives: tfidf for cpc and random patent sources, citation weights,
        number of features (ngrams) to find, pick=type of tfidf scoring, ngram multiplier,
        table path/filename
    Returns: Ranking of first six features with time option applied
    Side-effects: Prints and exports as excel a table as described above

    Postscript: A (better?) alternative may be to call method main in detect.py as a module,
        passing argparse args for base, focus, time, cite; then read the report files and combine
    """

    terms, scores_terms = tfidf.detect_popular_ngrams_in_corpus(
        number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
        pick=pick, time=False, citation_count_dict=None)
    base_df = pd.DataFrame(list(scores_terms)[:num_ngrams])
    base_df.columns = ['Score', 'Term']
    base_df['Rank'] = base_df.index
    base_df = base_df.reindex(columns=['Term', 'Score', 'Rank'])

    f_set_terms = tfidf.detect_popular_ngrams_in_corpus_excluding_common(
        tfidf_random,
        number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
        pick=pick, time=False,
        citation_count_dict=None)
    dict_freqs = dict([((p[0]), p[1]) for p in scores_terms if p[1] in f_set_terms])
    focus_scores_terms = tuple(dict_freqs.items())
    focus_df = pd.DataFrame(list(focus_scores_terms))[:num_ngrams]
    focus_df.columns = ['Focus Score', 'Term']
    focus_df['Focus Rank'] = focus_df.index
    focus_df = focus_df.reindex(columns=['Term', 'Focus Score', 'Focus Rank'])

    df = pd.merge(base_df, focus_df, how='outer')
    df['Diff Base to Focus Rank'] = df['Rank'] - df['Focus Rank']

    time_terms, time_scores_terms = tfidf.detect_popular_ngrams_in_corpus(
        number_of_ngrams_to_return=num_ngrams,
        pick=pick, time=True, citation_count_dict=None)
    time_df = pd.DataFrame(list(time_scores_terms))
    time_df.columns = ['Time Score', 'Term']
    time_df['Time Rank'] = time_df.index
    time_df = time_df.reindex(columns=['Term', 'Time Score', 'Time Rank'])

    df = pd.merge(df, time_df, how='outer')
    df['Diff Base to Time Rank'] = df['Rank'] - df['Time Rank']

    citation_terms, citation_scores_terms = tfidf.detect_popular_ngrams_in_corpus(
        number_of_ngrams_to_return=num_ngrams,
        pick=pick, time=False, citation_count_dict=citation_count_dict)
    citation_df = pd.DataFrame(list(citation_scores_terms))
    citation_df.columns = ['Citation Score', 'Term']
    citation_df['Citation Rank'] = citation_df.index
    citation_df = citation_df.reindex(columns=['Term', 'Citation Score', 'Citation Rank'])

    df = pd.merge(df, citation_df, how='outer')
    df['Diff Base to Citation Rank'] = df['Rank'] - df['Citation Rank']

    writer = pd.ExcelWriter(table_name, engine='xlsxwriter')
    df.to_excel(writer, 'Summary')
    base_df.to_excel(writer, 'Base')
    focus_df.to_excel(writer, 'Focus')
    time_df.to_excel(writer, 'Time')
    citation_df.to_excel(writer, 'Cite')
    writer.save()

    check_list = list(df['Time Rank'].iloc[:6])

    return check_list
