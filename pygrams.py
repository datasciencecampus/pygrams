import argparse
import bz2
import calendar
import json
import os
import pickle
import sys

import pandas as pd
from pandas import Timestamp, ExcelWriter
from tqdm import tqdm

from scripts.algorithms.term_focus import TermFocus
from scripts.algorithms.tfidf import LemmaTokenizer, TFIDF
from scripts.utils.argschecker import ArgsChecker
from scripts.utils.datesToPeriods import tfidf_with_dates_to_weekly_term_counts
from scripts.utils.pickle2df import PatentsPickle2DataFrame
from scripts.utils.table_output import table_output
from scripts.visualization.graphs.terms_graph import TermsGraph
from scripts.visualization.wordclouds.multicloudplot import MultiCloudPlot


# -fc="Communications,Leadership, IT systems"
# -ah=Comment -ds=comments_2017.xls -mn=2 -fc="Communications"

def choose_last_day(year_in, month_in):
    return str(calendar.monthrange(int(year_in), int(month_in))[1])


def year2pandas_latest_date(year_in, month_in):
    if year_in is None:
        return Timestamp.now()

    if month_in is None:
        return Timestamp(str(year_in) + '-12-31')

    year_string = str(year_in) + '-' + str(month_in) + '-' + choose_last_day(year_in, month_in)
    return Timestamp(year_string)


def year2pandas_earliest_date(year_in, month_in):
    if year_in is None:
        return Timestamp('2000-01-01')

    if month_in is None:
        return Timestamp(str(year_in) + '-01-01')

    year_string = str(year_in) + '-' + str(month_in) + '-01'
    return Timestamp(year_string)


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="create report, wordcloud, and fdg graph for free text in documents")

    parser.add_argument("-cpc", "--cpc_classification", default=None,
                        help="the desired cpc classification (for patents only)")
    parser.add_argument("-c", "--cite", default=False, action="store_true",
                        help="weight terms by citations (for patents only)")

    parser.add_argument("-f", "--focus", default=None, choices=['set', 'chi2', 'mutual'],
                        help="clean output from terms that appear in general; 'set': set difference, "
                             "'chi2': chi2 for feature importance, "
                             "'mutual': mutual information for feature importance")
    parser.add_argument("-ndl", "--normalize_doc_length", default=False, action="store_true",
                        help="normalize tf-idf scores by document length")
    parser.add_argument("-t", "--time", default=False, action="store_true", help="weight terms by time")

    parser.add_argument("-pt", "--path", default='data', help="the data path")
    parser.add_argument("-ih", "--id_header", default=None, help="the column name for the unique ID")
    parser.add_argument("-th", "--text_header", default='abstract', help="the column name for the free text")
    parser.add_argument("-dh", "--date_header", default=None, help="the column name for the date")
    parser.add_argument("-fc", "--filter_columns", default=None, help="list of columns to filter by")
    parser.add_argument("-fb", "--filter_by", default='union', choices=['union', 'intersection'],
                        help="options are <all> <any> defaults to any. Returns filter where all are 'Yes' "
                             "or any are 'Yes")

    parser.add_argument("-p", "--pick", default='sum', choices=['median', 'max', 'sum', 'avg'],
                        help="options are <median> <max> <sum> <avg>  defaults to sum. Average is over non zero values")
    parser.add_argument("-o", "--output", default='report',
                        choices=['fdg', 'wordcloud', 'report', 'table', 'tfidf', 'termcounts', 'all'],
                        help="options are: <fdg> <wordcloud> <report> <table> <tfidf> <termcounts> <all>")
    parser.add_argument("-j", "--json", default=False, action="store_true",
                        help="Output configuration as JSON file alongside output report")
    parser.add_argument("-yf", "--year_from", default=None,
                        help="The first year for the document cohort in YYYY format")
    parser.add_argument("-mf", "--month_from", default=None,
                        help="The first month for the document cohort in MM format")
    parser.add_argument("-yt", "--year_to", default=None, help="The last year for the document cohort in YYYY format")
    parser.add_argument("-mt", "--month_to", default=None, help="The last month for the document cohort in MM format")

    parser.add_argument("-np", "--num_ngrams_report", type=int, default=250,
                        help="number of ngrams to return for report")
    parser.add_argument("-nd", "--num_ngrams_wordcloud", type=int, default=250,
                        help="number of ngrams to return for wordcloud")
    parser.add_argument("-nf", "--num_ngrams_fdg", type=int, default=250,
                        help="number of ngrams to return for fdg graph")

    parser.add_argument("-ds", "--doc_source", default='USPTO-random-1000.pkl.bz2', help="the doc source to process")
    parser.add_argument("-fs", "--focus_source", default='USPTO-random-1000.pkl.bz2',
                        help="the doc source for the focus function")

    parser.add_argument("-mn", "--min_n", type=int, choices=[1, 2, 3], default=1, help="the minimum ngram value")
    parser.add_argument("-mx", "--max_n", type=int, choices=[1, 2, 3], default=3, help="the maximum ngram value")
    parser.add_argument("-mdf", "--max_document_frequency", type=float, default=0.05,
                        help="the maximum document frequency to contribute to TF/IDF")

    parser.add_argument("-rn", "--report_name", default=os.path.join('outputs', 'reports', 'report_tech.txt'),
                        help="report filename")
    parser.add_argument("-wn", "--wordcloud_name", default=os.path.join('outputs', 'wordclouds',
                                                                        'wordcloud_tech.png'),
                        help="wordcloud filename")
    parser.add_argument("-wt", "--wordcloud_title", default='tech terms', help="wordcloud title")

    parser.add_argument("-tn", "--table_name", default=os.path.join('outputs', 'table', 'table.xlsx'),
                        help="table filename")

    parser.add_argument("-nltk", "--nltk_path", default=None, help="custom path for NLTK data")

    args = parser.parse_args(command_line_arguments)
    return args

#not sure if this should be here. I have a feeling we may need to think of a minor refactoring, making sure things are on the right place
#especially if other devs jump in the project. Save some of their time for useful things :)
def get_tfidf(args, pickle_file_name, df=None, cpc=None):
    date_from = year2pandas_earliest_date(args.year_from, args.month_from)
    date_to = year2pandas_latest_date(args.year_to, args.month_to)
    if df is None or args.year_from is not None or args.year_to is not None:
        df = PatentsPickle2DataFrame(pickle_file_name, date_from=date_from, date_to=date_to).data_frame
    header_filter_cols = [x.strip() for x in args.filter_columns.split(",")] if args.filter_columns is not None else []
    header_lists = []
    doc_set = set([])
    if len(header_filter_cols) > 0:
        for header_idx in range(len(header_filter_cols)):
            header_list = df[header_filter_cols[header_idx]]
            header_idx_list = []
            for row_idx, value in enumerate(header_list):
                if value == 1 or value.lower() == 'yes':
                    header_idx_list.append(row_idx)
            header_lists.append(header_idx_list)
        doc_set = set(header_lists[0])

        for indices in header_lists[1:]:
            if args.filter_by == 'intersection':
                doc_set = doc_set.intersection(set(indices))
            else:
                doc_set = doc_set.union(set(indices))

    cpc_index_list=[]
    if cpc is not None:
        df = df.reset_index(drop=True)
        for index, row in tqdm(df.iterrows(), desc='Sifting documents for ' + cpc, unit='document', total=df.shape[0]):
            cpc_list = row['classifications_cpc']
            for cpc_item in cpc_list:
                if cpc_item.startswith(cpc):
                    cpc_index_list.append(index)
                    break
        if args.filter_by == 'intersection':
            doc_set = doc_set.intersection(set(cpc_index_list))
        else:
            doc_set = doc_set.union(set(cpc_index_list))

    return TFIDF(df, tokenizer=LemmaTokenizer(), ngram_range=(args.min_n, args.max_n), id_header=args.id_header,
                 text_header=args.text_header, date_header=args.date_header,
                 normalize_doc_length=args.normalize_doc_length,
                 max_document_frequency=args.max_document_frequency), doc_set


def load_citation_count_dict():
    citation_count_dict = pd.read_pickle(os.path.join('data/USPTO-citation-dictionary-family-pt1.pkl.bz2'))
    citation_count_dict_pt2 = pd.read_pickle(os.path.join('data/USPTO-citation-dictionary-family-pt2.pkl.bz2'))
    citation_count_dict.update(citation_count_dict_pt2)
    return citation_count_dict


def run_table(args, ngram_multiplier, tfidf, tfidf_random, citation_count_dict=None):
    if citation_count_dict is None:
        citation_count_dict = load_citation_count_dict()

    num_ngrams = max(args.num_ngrams_report, args.num_ngrams_wordcloud)
    print(f'Writing table to {args.table_name}')
    writer = ExcelWriter(args.table_name, engine='xlsxwriter')

    table_output(tfidf, tfidf_random, num_ngrams, args, ngram_multiplier, writer,
                 citation_count_dict=citation_count_dict)


# TODO: common interface wrapper class, hence left citation_count_dict refs
def run_report(args, ngram_multiplier, tfidf, tfidf_random=None, wordclouds=False,
               docs_set=None, citation_count_dict=None):
    num_ngrams = max(args.num_ngrams_report, args.num_ngrams_wordcloud)

    tfocus = TermFocus(tfidf, tfidf_random, id_header=args.id_header,
                       text_header=args.text_header, date_header=args.date_header)
    dict_freqs, focus_set_terms, _ = tfocus.detect_and_focus_popular_ngrams(args, citation_count_dict, ngram_multiplier,
                                                                            num_ngrams, docs_set=docs_set)
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
    return dict_freqs


def run_fdg(dict_freq_in, tf_idf, report_name, num_ngrams):
    graph = TermsGraph(dict_freq_in, tf_idf)
    graph.save_graph_report(report_name, num_ngrams)
    graph.save_graph("key-terms", 'data')


def write_config_to_json(args, doc_pickle_file_name):
    doc_pickle_file_name = os.path.abspath(doc_pickle_file_name)
    report_file_name = os.path.abspath(args.report_name)
    json_file_name = os.path.splitext(report_file_name)[0] + '.json'

    month_from = args.month_from if args.month_from is not None else '01'
    month_to = args.month_to if args.month_to is not None else '12'
    year_from = args.year_from if args.year_from is not None else '2000'
    year_to = args.year_to if args.year_to is not None else str(Timestamp.now().year)

    json_data = {
        'paths': {
            'data': doc_pickle_file_name,
            'tech_report': report_file_name
        },
        'month_year': {
            'from': month_from + '_' + year_from,
            'to': month_to + '_' + year_to
        },
        'parameters': {
            'pick': args.pick,
            'time': args.time,
            'focus': args.focus
        }
    }

    with open(json_file_name, 'w') as json_file:
        json.dump(json_data, json_file)


def output_tfidf(tfidf_base_filename, tfidf):
    try:
        dates = tfidf.dates
        document_week_dates = [iso_date[0] * 100 + iso_date[1] for iso_date in
                               [d.isocalendar() for d in dates]]
    except (KeyError, ValueError):
        document_week_dates = [None] * len(tfidf.feature_names)

    try:
        doc_ids = tfidf.doc_ids
    except KeyError:
        doc_ids = [None] * len(tfidf.feature_names)

    tfidf_data = [tfidf.tfidf_matrix, tfidf.feature_names, document_week_dates, doc_ids]
    tfidf_filename = os.path.join('outputs', 'tfidf', tfidf_base_filename + '-tfidf.pkl.bz2')
    os.makedirs(os.path.dirname(tfidf_filename), exist_ok=True)
    with bz2.BZ2File(tfidf_filename, 'wb') as pickle_file:
        pickle.dump(tfidf_data, pickle_file)

    term_present_matrix = tfidf.tfidf_matrix > 0
    term_present_data = [term_present_matrix, tfidf.feature_names, document_week_dates, doc_ids]
    term_present_filename = os.path.join('outputs', 'tfidf', tfidf_base_filename + '-term_present.pkl.bz2')
    os.makedirs(os.path.dirname(term_present_filename), exist_ok=True)
    with bz2.BZ2File(term_present_filename, 'wb') as pickle_file:
        pickle.dump(term_present_data, pickle_file)


def output_term_counts(tfidf_base_filename, tfidf):
    try:
        dates = tfidf.dates
        document_week_dates = [iso_date[0] * 100 + iso_date[1] for iso_date in
                               [d.isocalendar() for d in dates]]
    except ValueError:
        dates = [None] * len(tfidf.doc_ids)

    term_counts_per_week, number_of_documents_per_week, week_iso_dates = tfidf_with_dates_to_weekly_term_counts(
        tfidf.tfidf_matrix, document_week_dates)

    term_counts_data = [term_counts_per_week, tfidf.feature_names, number_of_documents_per_week, week_iso_dates]
    term_counts_filename = os.path.join('outputs', 'termcounts', tfidf_base_filename + '-term_counts.pkl.bz2')
    os.makedirs(os.path.dirname(term_counts_filename), exist_ok=True)
    with bz2.BZ2File(term_counts_filename, 'wb') as pickle_file:
        pickle.dump(term_counts_data, pickle_file)


def main(supplied_args):
    paths = [os.path.join('outputs', 'reports'), os.path.join('outputs', 'wordclouds'),
             os.path.join('outputs', 'table')]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    args = get_args(supplied_args)
    args_default = get_args([])
    argscheck = ArgsChecker(args, args_default)
    argscheck.checkargs()

    doc_source_file_name = os.path.join(args.path, args.doc_source)

    df = None
    if doc_source_file_name.endswith('.pkl.bz2'):
        df = pd.read_pickle(doc_source_file_name)
    elif doc_source_file_name.endswith('.xls'):
        df = pd.read_excel(doc_source_file_name)
    elif doc_source_file_name.endswith('.csv'):
        df = pd.read_csv(doc_source_file_name, engine='python', error_bad_lines=False, nrows=10000)
    elif doc_source_file_name.endswith('.xlsx'):
        df = pd.read_excel(doc_source_file_name)

    argscheck.checkdf(df)

    if isinstance(args.filter_columns, type(None)):
        docs_set = None
    else:
        filter_headers = args.filter_columns.split(',')
        filter_headers = [header.strip() for header in filter_headers]
        filter_by = args.filter_by

        filter_df = df.copy()
        filter_df = filter_df[filter_headers]

        for column in filter_df:
            filter_df[column] = filter_df[column].replace({'No': 0, 'Yes': 1})
        filter_df['filter'] = filter_df.sum(axis=1)

        if filter_by == 'any':
            docs_set = filter_df[filter_df['filter'] > 0].index.values.tolist()
        else:
            docs_set = filter_df[filter_df['filter'] == filter_df.shape[1] - 1].index.values.tolist()

    if args.json:
        write_config_to_json(args, doc_source_file_name)

    if args.nltk_path:
        import nltk
        nltk.data.path.append(args.nltk_path)

    tfidf, doc_set = get_tfidf(args, doc_source_file_name, df=df, cpc=args.cpc_classification)

    newtfidf = None
    if args.focus or args.output == 'table':
        path2 = os.path.join('data', args.focus_source)
        newtfidf, _ = get_tfidf(args, path2, None)

    citation_count_dict = None
    if args.cite:
        citation_count_dict = load_citation_count_dict()

    out = args.output
    ngram_multiplier = 4

    if out != 'tfidf':
        wordclouds_flag = (out == 'wordcloud')
        dict_freqs = run_report(args, ngram_multiplier, tfidf, newtfidf, wordclouds=wordclouds_flag, docs_set=doc_set,
                                citation_count_dict=citation_count_dict)

    if out == 'table' or out == 'all':
        run_table(args, ngram_multiplier, tfidf, newtfidf, citation_count_dict=citation_count_dict)

    if out == 'fdg' or out == 'all':
        num_ngrams = args.num_ngrams_report
        freqs_list = list(dict_freqs.items())[:num_ngrams]
        run_fdg(freqs_list, tfidf, args.report_name, args.num_ngrams_report)

    if out == 'tfidf' or out == 'all':
        output_tfidf(args.doc_source, tfidf)

    if out == 'termcounts' or out == 'all':
        output_term_counts(args.doc_source, tfidf)


if __name__ == '__main__':
    main(sys.argv[1:])
