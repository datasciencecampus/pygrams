import pandas as pd

from scripts.algorithms import term_focus


def table_output(tfidf, tfidf_random, citation_count_dict, num_ngrams, pick, ngram_multiplier, time, focus, writer):
    """
    Description: Creates a table showing changes in feature rankings with focus, time, and cite options
    Receives: tfidf for cpc and random patent sources, citation weights,
        number of features (ngrams) to find, pick=type of tfidf scoring, ngram multiplier,
        table path/filename
    Returns: Ranking of first six features with time option applied (for unit test purposes)
    Side-effects: Exports in excel format a table as described above

    Postscript: A (better?) alternative may be to call method main in detect.py as a module,
        passing argparse args for base, focus, time, cite; then read the report files and combine
    """

    dict_freqs, focus_set_terms, scores_terms = term_focus.detect_and_focus_popular_ngrams(pick, time, focus, citation_count_dict, ngram_multiplier, num_ngrams,
                                                                                      tfidf, tfidf_random)

    base_df = pd.DataFrame(list(scores_terms)[:num_ngrams])
    base_df.columns = ['Score', 'Term']
    base_df['Rank'] = base_df.index
    base_df = base_df.reindex(columns=['Term', 'Score', 'Rank'])

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

    df.to_excel(writer, 'Summary')
    base_df.to_excel(writer, 'Base')
    focus_df.to_excel(writer, 'Focus')
    time_df.to_excel(writer, 'Time')
    citation_df.to_excel(writer, 'Cite')
    writer.save()
