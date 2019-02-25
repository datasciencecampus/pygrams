import argparse
import os
import sys

from scripts.pipeline import Pipeline
from scripts.utils.argschecker import ArgsChecker


# -fc="Communications,Leadership, IT systems"
# -ah=Comment -ds=comments_2017.xls -mn=2 -fc="Communications"


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
    parser.add_argument("-o", "--output", default=['report'], nargs='*',
                        choices=['graph', 'wordcloud', 'report', 'table', 'tfidf', 'termcounts'],
                        help="options are: <graph> <wordcloud> <report> <table> <tfidf> <termcounts>;"
                             " note that this can be defined multiple times to get more than one output")
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
    parser.add_argument("-nf", "--num_ngrams_fdg", type=int, default=50,
                        help="number of ngrams to return for fdg graph")

    parser.add_argument("-ds", "--doc_source", default='USPTO-random-1000.pkl.bz2', help="the doc source to process")
    parser.add_argument("-fs", "--focus_source", default='USPTO-random-1000.pkl.bz2',
                        help="the doc source for the focus function")

    parser.add_argument("-mn", "--min_n", type=int, choices=[1, 2, 3], default=2, help="the minimum ngram value")
    parser.add_argument("-mx", "--max_n", type=int, choices=[1, 2, 3], default=3, help="the maximum ngram value")
    parser.add_argument("-mdf", "--max_document_frequency", type=float, default=0.05,
                        help="the maximum document frequency to contribute to TF/IDF")

    parser.add_argument("-on", "--outputs_name", default='out', help="outputs filename")

    parser.add_argument("-wt", "--wordcloud_title", default='tech terms', help="wordcloud title")

    parser.add_argument("-tn", "--table_name", default=os.path.join('outputs', 'table', 'table.xlsx'),
                        help="table filename")

    parser.add_argument("-nltk", "--nltk_path", default=None, help="custom path for NLTK data")

    args = parser.parse_args(command_line_arguments)
    return args


def main(supplied_args):
    paths = [os.path.join('outputs', 'reports'), os.path.join('outputs', 'wordclouds'),
             os.path.join('outputs', 'table'), os.path.join('outputs', 'tfidf_wrapper')]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    args = get_args(supplied_args)
    args_default = get_args([])
    argscheck = ArgsChecker(args, args_default)
    argscheck.checkargs()

    doc_weights_dict = argscheck.get_mask_dict()

    doc_source_file_name = os.path.join(args.path, args.doc_source)
    tfidf_wrapper_filename = os.path.join('outputs', 'tfidf_wrapper', 'tfidf_wrapper.pickle')
    pickled_tf_idf = os.path.isfile(tfidf_wrapper_filename)
    pipeline = Pipeline(doc_source_file_name, filter_columns=args.filter_columns, pick_method=args.pick,
                        max_n=args.max_n, min_n=args.min_n, normalize_rows=args.normalize_doc_length, filter_by=args.filter_by,
                        nterms=args.num_ngrams_report, text_header=args.text_header, max_df=args.max_document_frequency,
                        term_counts=('termcounts' in args.output), dates_header=args.date_header, pickled_tf_idf=pickled_tf_idf,
                        tfidf_wrapper_filename=tfidf_wrapper_filename)

    pipeline.output(args.output, wordcloud_title=args.wordcloud_title, outname=args.outputs_name, nterms=50)


if __name__ == '__main__':
    main(sys.argv[1:])
