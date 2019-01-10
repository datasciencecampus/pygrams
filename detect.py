import argparse
import bz2
import json
import os
import pickle
import sys

from pandas import Timestamp, read_pickle, ExcelWriter

from scripts import FilePaths
from scripts.algorithms.term_focus import TermFocus
from scripts.algorithms.tfidf import LemmaTokenizer, TFIDF
from scripts.utils.datesToPeriods import tfidf_with_dates_to_weekly_term_counts
from scripts.utils.pickle2df import PatentsPickle2DataFrame
from scripts.utils.table_output import table_output
from scripts.visualization.graphs.terms_graph import TermsGraph
from scripts.visualization.wordclouds.multicloudplot import MultiCloudPlot


def year2pandas_latest_date(year_in):
    if year_in == 0:
        return Timestamp.now()

    year_string = str(year_in) + '-12-31'
    return Timestamp(year_string)


def year2pandas_earliest_date(year_in):
    year_string = str(year_in) + '-01-01'
    return Timestamp(year_string)


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="create report, wordcloud, and fdg graph for patent texts")

    parser.add_argument("-f", "--focus", default=None, choices=['set', 'chi2', 'mutual'],
                        help="clean output from terms that appear in general; 'set': set difference, "
                             "'chi2': chi2 for feature importance, "
                             "'mutual': mutual information for feature importance")
    parser.add_argument("-c", "--cite", default=False, action="store_true", help="weight terms by citations")
    parser.add_argument("-t", "--time", default=False, action="store_true", help="weight terms by time")

    parser.add_argument("-p", "--pick", default='sum', choices=['median', 'max', 'sum', 'avg'],
                        help="options are <median> <max> <sum> <avg>  defaults to sum. Average is over non zero values")
    parser.add_argument("-o", "--output", default='report',
                        choices=['fdg', 'wordcloud', 'report', 'table', 'tfidf', 'termcounts', 'all'],
                        help="options are: <fdg> <wordcloud> <report> <table> <tfidf> <termcounts> <all>")
    parser.add_argument("-j", "--json", default=False, action="store_true",
                        help="Output configuration as JSON file alongside output report")
    parser.add_argument("-yf", "--year_from", type=int, default=2000, help="The first year for the patent cohort")
    parser.add_argument("-yt", "--year_to", type=int, default=0, help="The last year for the patent cohort (0 is now)")

    parser.add_argument("-np", "--num_ngrams_report", type=int, default=250,
                        help="number of ngrams to return for report")
    parser.add_argument("-nd", "--num_ngrams_wordcloud", type=int, default=250,
                        help="number of ngrams to return for wordcloud")
    parser.add_argument("-nf", "--num_ngrams_fdg", type=int, default=50,
                        help="number of ngrams to return for fdg graph")

    parser.add_argument("-ps", "--patent_source", default='USPTO-random-1000', help="the patent source to process")
    parser.add_argument("-fs", "--focus_source", default='USPTO-random-10000',
                        help="the patent source for the focus function")

    parser.add_argument("-mn", "--min_n", type=int, choices=[1, 2, 3], default=2, help="the minimum ngram value")
    parser.add_argument("-mx", "--max_n", type=int, choices=[1, 2, 3], default=3, help="the maximum ngram value")
    parser.add_argument("-mdf", "--max_document_frequency", type=float, default=0.3,
                        help="the maximum document frequency to contribute to TF/IDF")

    parser.add_argument("-rn", "--report_name", default=os.path.join('outputs', 'reports', 'report_tech.txt'),
                        help="report filename")
    parser.add_argument("-wn", "--wordcloud_name", default=os.path.join('outputs', 'wordclouds', 'wordcloud_tech.png'),
                        help="wordcloud filename")
    parser.add_argument("-wt", "--wordcloud_title", default='tech terms', help="wordcloud title")

    parser.add_argument("-tn", "--table_name", default=os.path.join('outputs', 'table', 'table.xlsx'),
                        help="table filename")
    parser.add_argument("-cpc", "--cpc_classification", default=None, help="the desired cpc classification")

    parser.add_argument("-nltk", "--nltk_path", default=None, help="custom path for NLTK data")

    args = parser.parse_args(command_line_arguments)
    return args


def checkargs(args):
    app_exit = False
    if args.year_to != 0:
        if args.year_from >= args.year_to:
            print("year_from must be less than year_to")
            app_exit = True

    if args.min_n > args.max_n:
        print("minimum ngram count should be less or equal to higher ngram count")
        app_exit = True

    if args.num_ngrams_wordcloud <= 20:
        print("at least 20 ngrams needed for wordcloud")
        app_exit = True

    if args.num_ngrams_report <= 10:
        print("at least 10 ngrams needed for report")
        app_exit = True

    if args.output == 'table' or args.output == 'all':
        if args.focus is None:
            print('define a focus before requesting table (or all) output')
            app_exit = True

    if app_exit:
        exit(0)


def check_cpc_between_years(args, df):
    cpc = args.cpc_classification if args.cpc_classification is not None else "all"
    lendf = len(df)
    if lendf <= 100:
        print(str(lendf) + " records found for cpc=" + cpc + ", between " + str(args.year_from) + " and " + str(
            args.year_to))
        print(
            "Not sufficient for tf-idf analysis. Please change parameters to raise the resulting patents cohort count")
        exit(0)


def get_tfidf(args, pickle_file_name, cpc):
    date_from = year2pandas_earliest_date(args.year_from)
    date_to = year2pandas_latest_date(args.year_to)

    df = PatentsPickle2DataFrame(pickle_file_name, classification=cpc, date_from=date_from, date_to=date_to).data_frame
    check_cpc_between_years(args, df)
    return TFIDF(df, tokenizer=LemmaTokenizer(), ngram_range=(args.min_n, args.max_n),
                 max_document_frequency=args.max_document_frequency)


def load_citation_count_dict():
    citation_count_dict = read_pickle(FilePaths.us_patents_citation_dictionary_1of2_pickle_name)
    citation_count_dict_pt2 = read_pickle(FilePaths.us_patents_citation_dictionary_2of2_pickle_name)
    citation_count_dict.update(citation_count_dict_pt2)
    return citation_count_dict


def run_table(args, ngram_multiplier, tfidf, tfidf_random, citation_count_dict):
    if citation_count_dict is None:
        citation_count_dict = load_citation_count_dict()

    num_ngrams = max(args.num_ngrams_report, args.num_ngrams_wordcloud)

    print(f'Writing table to {args.table_name}')
    writer = ExcelWriter(args.table_name, engine='xlsxwriter')

    table_output(tfidf, tfidf_random, num_ngrams, args.pick, ngram_multiplier, args.time,
                 args.focus, writer, citation_count_dict=citation_count_dict)


def run_report(args, ngram_multiplier, tfidf, tfidf_random=None, wordclouds=False, citation_count_dict=None):
    num_ngrams = max(args.num_ngrams_report, args.num_ngrams_wordcloud)

    tfocus = TermFocus(tfidf, tfidf_random)
    dict_freqs, focus_set_terms, _ = tfocus.detect_and_focus_popular_ngrams(args.pick, args.time, args.focus,
                                                                            citation_count_dict, ngram_multiplier,
                                                                            num_ngrams)

    with open(args.report_name, 'w') as file:
        counter = 1
        for score, term in dict_freqs.items():
            file.write(f' {term:30} {score:f}\n')
            print(f'{counter}. {term:30} {score:f}')
            counter += 1
            if counter > args.num_ngrams_report:
                break

    if wordclouds:
        doc_all = ' '.join(focus_set_terms)
        wordcloud = MultiCloudPlot(doc_all, freqsin=dict_freqs, max_words=args.num_ngrams_wordcloud)
        wordcloud.plot_cloud(args.wordcloud_title, args.wordcloud_name)


def run_fdg(dict_freq_in, tf_idf, args):
    num_ngrams = args.num_ngrams_report
    graph = TermsGraph(list(dict_freq_in.items())[:num_ngrams], tf_idf)
    graph.save_graph_report(args)
    graph.save_graph("key-terms", 'data')


def write_config_to_json(args, patent_pickle_file_name):
    patent_pickle_file_name = os.path.abspath(patent_pickle_file_name)
    report_file_name = os.path.abspath(args.report_name)
    json_file_name = os.path.splitext(report_file_name)[0] + '.json'

    json_data = {
        'paths': {
            'data': patent_pickle_file_name,
            'tech_report': report_file_name
        },
        'year': {
            'from': args.year_from,
            'to': args.year_to
        },
        'parameters': {
            'cpc': '' if args.cpc_classification is None else args.cpc_classification,
            'pick': args.pick,
            'time': args.time,
            'cite': args.cite,
            'focus': args.focus
        }
    }

    with open(json_file_name, 'w') as json_file:
        json.dump(json_data, json_file)


def output_tfidf(tfidf_base_filename, tfidf, ngram_multiplier, num_ngrams, pick, time, citation_count_dict):
    terms, ngrams_scores_tuple, tfidf_matrix = tfidf.detect_popular_ngrams_in_docs_set(
        number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
        pick=pick, time=time,
        citation_count_dict=citation_count_dict)

    publication_week_dates = [iso_date[0] * 100 + iso_date[1] for iso_date in
                              [d.isocalendar() for d in tfidf.publication_dates]]

    tfidf_data = [tfidf_matrix, tfidf.feature_names, publication_week_dates, tfidf.patent_ids]
    tfidf_filename = os.path.join('outputs', 'tfidf', tfidf_base_filename + '-tfidf.pkl.bz2')
    os.makedirs(os.path.dirname(tfidf_filename), exist_ok=True)
    with bz2.BZ2File(tfidf_filename, 'wb') as pickle_file:
        pickle.dump(tfidf_data, pickle_file)


def output_term_counts(tfidf_base_filename, tfidf, ngram_multiplier, num_ngrams, pick, time, citation_count_dict):
    terms, ngrams_scores_tuple, tfidf_matrix = tfidf.detect_popular_ngrams_in_corpus(
        number_of_ngrams_to_return=ngram_multiplier * num_ngrams,
        pick=pick, time=time,
        citation_count_dict=citation_count_dict)

    publication_week_dates = [iso_date[0] * 100 + iso_date[1] for iso_date in
                              [d.isocalendar() for d in tfidf.publication_dates]]

    term_counts_per_week, number_of_patents_per_week, week_iso_dates = tfidf_with_dates_to_weekly_term_counts(
        tfidf_matrix, publication_week_dates)

    term_counts_data = [term_counts_per_week, tfidf.feature_names, number_of_patents_per_week, week_iso_dates]
    term_counts_filename = os.path.join('outputs', 'termcounts', tfidf_base_filename + '-term_counts.pkl.bz2')
    os.makedirs(os.path.dirname(term_counts_filename), exist_ok=True)
    with bz2.BZ2File(term_counts_filename, 'wb') as pickle_file:
        pickle.dump(term_counts_data, pickle_file)


def main():
    paths = [os.path.join('outputs', 'reports'), os.path.join('outputs', 'wordclouds'),
             os.path.join('outputs', 'table')]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    args = get_args(sys.argv[1:])
    checkargs(args)

    patent_pickle_file_name = os.path.join('data', args.patent_source + ".pkl.bz2")

    if args.json:
        write_config_to_json(args, patent_pickle_file_name)

    if args.nltk_path:
        import nltk
        nltk.data.path.append(args.nltk_path)

    tfidf = get_tfidf(args, patent_pickle_file_name, args.cpc_classification)

    newtfidf = None
    if args.focus or args.output == 'table':
        path2 = os.path.join('data', args.focus_source + ".pkl.bz2")
        newtfidf = get_tfidf(args, path2, None)

    citation_count_dict = None
    if args.cite:
        citation_count_dict = load_citation_count_dict()

    out = args.output

    ngram_multiplier = 4

    if out == 'report':
        run_report(args, ngram_multiplier, tfidf, newtfidf, citation_count_dict=citation_count_dict)
    elif out == 'wordcloud' or out == 'all':
        run_report(args, ngram_multiplier, tfidf, newtfidf, wordclouds=True, citation_count_dict=citation_count_dict)

    if out == 'table' or out == 'all':
        run_table(args, ngram_multiplier, tfidf, newtfidf, citation_count_dict)

    if out == 'fdg' or out == 'all':
        run_fdg(args, tfidf, newtfidf)

    if out == 'tfidf' or out == 'all':
        output_tfidf(args.patent_source, tfidf, ngram_multiplier, args.num_ngrams_report, args.pick, args.time,
                     citation_count_dict=citation_count_dict)

    if out == 'termcounts' or out == 'all':
        output_term_counts(args.patent_source, tfidf, ngram_multiplier, args.num_ngrams_report, args.pick, args.time,
                           citation_count_dict=citation_count_dict)


if __name__ == '__main__':
    main()
