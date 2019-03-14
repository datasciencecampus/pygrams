import argparse
import os
import sys

from scripts.pipeline import Pipeline
from scripts.utils.argschecker import ArgsChecker
from scripts.utils.pygrams_exception import PygramsException


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
                        help="options are <union> <intersection> defaults to union. Returns filter where all are 'Yes' "
                             "or any are 'Yes' in the defined --filter_columns")

    parser.add_argument("-p", "--pick", default='sum', choices=['median', 'max', 'sum', 'avg'],
                        help="options are <median> <max> <sum> <avg>  defaults to sum. Everything is computed over "
                             "non zero values")
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

    parser.add_argument("-ds", "--doc_source", default='USPTO-random-1000.pkl.bz2', help="the document source to process")
    parser.add_argument("-it", "--input_tfidf", default=None,
                        help="pickled TFIDF output instead of processing a document source")
    parser.add_argument("-fs", "--focus_source", default='USPTO-random-1000.pkl.bz2',
                        help="the document source for the focus function")

    parser.add_argument("-mn", "--min_n", type=int, choices=[1, 2, 3], default=1, help="the minimum ngram value")
    parser.add_argument("-mx", "--max_n", type=int, choices=[1, 2, 3], default=3, help="the maximum ngram value")
    parser.add_argument("-mdf", "--max_document_frequency", type=float, default=0.05,
                        help="the maximum document frequency to contribute to TF/IDF")

    parser.add_argument("-on", "--outputs_name", default='out', help="outputs filename")

    parser.add_argument("-wt", "--wordcloud_title", default='Popular Terms', help="wordcloud title")

    parser.add_argument("-tn", "--table_name", default=os.path.join('outputs', 'table', 'table.xlsx'),
                        help="table filename")

    parser.add_argument("-nltk", "--nltk_path", default=None, help="custom path for NLTK data")

    args = parser.parse_args(command_line_arguments)
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
    outputs = args.output[:]
    outputs.append('json_config')
    docs_mask_dict = argscheck.get_docs_mask_dict()
    terms_mask_dict = argscheck.get_terms_mask_dict()

    doc_source_file_name = os.path.join(args.path, args.doc_source)
    if args.input_tfidf is None:
        pickled_tf_idf_path = None
    else:
        pickled_tf_idf_path = os.path.join(args.path, args.input_tfidf)

    pipeline = Pipeline(doc_source_file_name, docs_mask_dict, pick_method=args.pick,
                        ngram_range=(args.min_n, args.max_n), normalize_rows=args.normalize_doc_length,
                        text_header=args.text_header, max_df=args.max_document_frequency,
                        term_counts=('termcounts' in args.output),
                        pickled_tf_idf_file_name=pickled_tf_idf_path, tfidf_output='tfidf' in args.output,
                        output_name=args.outputs_name)

    if 'tfidf' in outputs:
        outputs.remove('tfidf')

    pipeline.output(outputs, wordcloud_title=args.wordcloud_title, outname=args.outputs_name, nterms=50)


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except PygramsException as err:
        print(f"pyGrams error: {err.message}")
