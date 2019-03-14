import argparse
import os
import sys

from scripts.pipeline import Pipeline
from scripts.utils.argschecker import ArgsChecker
from scripts.utils.pygrams_exception import PygramsException

'''
Link pygrams help (get_args) to main sequence, to pipeline sequence, and to manual (sequence).

Keep: cpc, ndl, time, th, dh, fc = 1 or yes, fb, filter rows by, p, tfidf, term counts - explain
Report always, default None, Dates = defaults data start, today, YYYY/MM/DD, two options
Num_grams, ds, Min_n -> min_grams, Output files names, timestamp? Windows/Mac, nltk

Suppress: cite, focus, pt path (put data in data folder), ih, table, Son - always save, fs, Table_name

Reorder arguments according to pipeline (/manual)

'''


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="extract popular n-grams (words or short phrases)"
                                                 " from a corpus of documents",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overridng of arguments

    # suppressed:________________________________________
    parser.add_argument("-ih", "--id_header", default=None, help="the column name for the unique ID")
    parser.add_argument("-c", "--cite", default=False, action="store_true",
                        help="weight terms by citations (for patents only)")
    parser.add_argument("-pt", "--path", default='data', help="the data path")

    # Focus source and function
    parser.add_argument("-f", "--focus", default=None, choices=['set', 'chi2', 'mutual'],
                        help="clean output from terms that appear in general; 'set': set difference, "
                             "'chi2': chi2 for feature importance, "
                             "'mutual': mutual information for feature importance")
    parser.add_argument("-fs", "--focus_source", default='USPTO-random-1000.pkl.bz2',
                        help="the document source for the focus function")
    parser.add_argument("-tn", "--table_name", default=os.path.join('outputs', 'table', 'table.xlsx'),
                        help="table filename")

    parser.add_argument("-j", "--json", default=True, action="store_true",
                        help="Output configuration as JSON file alongside output report")
    # __________________________________________________

    # Document source
    parser.add_argument("-ds", "--doc_source", default='USPTO-random-1000.pkl.bz2', help="the document source to process")

    # Document column header names
    parser.add_argument("-th", "--text_header", default='abstract', help="the column name for the free text")
    parser.add_argument("-dh", "--date_header", default=None, help="the column name for the date")

    # Word filters
    parser.add_argument("-fc", "--filter_columns", default=None,
                        help="list of columns with binary entries by which to filter the rows")
    parser.add_argument("-fb", "--filter_by", default='union', choices=['union', 'intersection'],
                        help="Returns filter where all are 'Yes' or '1'"
                             "or any are 'Yes' or '1' in the defined --filter_columns")

    # Time filters
    parser.add_argument("-df", "--date_from", default=None,
                        help="The first date for the document cohort in YYYY/MM/DD format")
    parser.add_argument("-dt", "--date_to", default=None,
                        help="The last date for the document cohort in YYYY/MM/DD format")

    # TF-IDF PARAMETERS

    # ngrams selection
    parser.add_argument("-mn", "--min_ngrams", type=int, choices=[1, 2, 3], default=1, help="the minimum ngram value")
    parser.add_argument("-mx", "--max_ngrams", type=int, choices=[1, 2, 3], default=3, help="the maximum ngram value")

    # maximum document frequency
    parser.add_argument("-mdf", "--max_document_frequency", type=float, default=0.05,
                        help="the maximum document frequency to contribute to TF/IDF")

    # tf-idf score mechanics
    parser.add_argument("-p", "--pick", default='sum', choices=['median', 'max', 'sum', 'avg'],
                        help="Everything is computed over "
                             "non zero values")

    # Normalize tf-idf scores by document length
    parser.add_argument("-ndl", "--normalize_doc_length", default=False, action="store_true",
                        help="normalize tf-idf scores by document length")

    # Time weighting
    parser.add_argument("-t", "--time", default=False, action="store_true", help="weight terms by time")

    # OUTPUT PARAMETERS
    # select outputs
    parser.add_argument("-o", "--output", default=['report'], nargs='*',
                        choices=['graph', 'wordcloud', 'report', 'tfidf', 'termcounts'],  # suppress table output option
                        # choices=['graph', 'wordcloud', 'report', 'table', 'tfidf', 'termcounts'],
                        help="Note that this can be defined multiple times to get more than one output. "
                             "termcounts represents the term frequency component of tfidf")

    # file names etc.
    parser.add_argument("-on", "--outputs_name", default='out', help="outputs filename")
    parser.add_argument("-wt", "--wordcloud_title", default='Popular Terms', help="wordcloud title")

    parser.add_argument("-nltk", "--nltk_path", default=None, help="custom path for NLTK data")

    # number of ngrams reported
    parser.add_argument("-np", "--num_ngrams_report", type=int, default=250,
                        help="number of ngrams to return for report")
    parser.add_argument("-nd", "--num_ngrams_wordcloud", type=int, default=250,
                        help="number of ngrams to return for wordcloud")
    parser.add_argument("-nf", "--num_ngrams_fdg", type=int, default=250,
                        help="number of ngrams to return for fdg graph")

    # PATENT SPECIFIC SUPPORT
    parser.add_argument("-cpc", "--cpc_classification", default=None,
                        help="the desired cpc classification (for patents only)")

    options_suppressed_in_help = [
        "-ih", "--id_header"
        "-c", "--cite",
        "-f", "--focus",
        "-pt", "--path",
        "-ih", "--id_header",
        "-fs", "--focus_source",
        "-tn", "--table_name",
        "-cpc", "--cpc_classification",
        "-z", "--zzz"
        ]

    for options in options_suppressed_in_help:
        parser.add_argument(options, help=argparse.SUPPRESS)

    args = parser.parse_args(command_line_arguments)
    # need to add non None defaults back in if they are required
    args.path = 'data'
    return args


def main(supplied_args):
    paths = [os.path.join('outputs', 'reports'), os.path.join('outputs', 'wordclouds'),
             os.path.join('outputs', 'table')]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    args = get_args(supplied_args)
    args_default = get_args([])
    argscheck = ArgsChecker(args, args_default)
    argscheck.checkargs()
    outputs = args.output
    outputs.append('json_config')
    docs_mask_dict = argscheck.get_docs_mask_dict()
    terms_mask_dict = argscheck.get_terms_mask_dict()

    doc_source_file_name = os.path.join(args.path, args.doc_source)
    pipeline = Pipeline(doc_source_file_name, docs_mask_dict,  pick_method=args.pick,
                        ngram_range=(args.min_ngrams, args.max_ngrams), normalize_rows=args.normalize_doc_length,
                        text_header=args.text_header, max_df=args.max_document_frequency,
                        term_counts=('termcounts' in args.output))

    pipeline.output(args.output, wordcloud_title=args.wordcloud_title, outname=args.outputs_name, nterms=50)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except PygramsException as err:
        print(f"pyGrams error: {err.message}")
