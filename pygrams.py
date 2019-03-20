import argparse
import os
import sys

from scripts.pipeline import Pipeline
from scripts.utils.argschecker import ArgsChecker
from scripts.utils.pygrams_exception import PygramsException

predictor_names = ['All', 'Naive', 'Linear', 'Quadratic', 'Cubic', 'ARIMA', 'Holt-Winters',
                   'LSTM-multiLA-stateful', 'LSTM-multiLA-stateless',
                   'LSTM-1LA-stateful', 'LSTM-1LA-stateless',
                   'LSTM-multiM-1LA-stateful', 'LSTM-multiM-1LA-stateless']


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="extract popular n-grams (words or short phrases)"
                                                 " from a corpus of documents",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overridng of arguments

    # suppressed:________________________________________
    parser.add_argument("-tc", "--term-counts", default=False,  action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-ih", "--id_header", default=None, help=argparse.SUPPRESS)
    parser.add_argument("-c", "--cite", default=False, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-pt", "--path", default='data', help=argparse.SUPPRESS)

    # Focus source and function
    parser.add_argument("-f", "--focus", default=None, choices=['set', 'chi2', 'mutual'],
                        help=argparse.SUPPRESS)
    parser.add_argument("-fs", "--focus_source", default='USPTO-random-1000.pkl.bz2', help=argparse.SUPPRESS)
    parser.add_argument("-tn", "--table_name", default=os.path.join('outputs', 'table', 'table.xlsx'), help=argparse.SUPPRESS)

    parser.add_argument("-j", "--json", default=True, action="store_true",
                        help=argparse.SUPPRESS)
    # tf-idf score mechanics
    parser.add_argument("-p", "--pick", default='sum', choices=['median', 'max', 'sum', 'avg'],
                        help=argparse.SUPPRESS)
    parser.add_argument("-tst", "--test", default=False, action="store_true", help=argparse.SUPPRESS)
    # end __________________________________________________

    # Input files
    parser.add_argument("-ds", "--doc_source", default='USPTO-random-1000.pkl.bz2',
                        help="the document source to process")
    parser.add_argument("-it", "--input_tfidf", default=None,
                        help="Load a pickled TFIDF output instead of creating TFIDF by processing a document source")

    # Document column header names
    parser.add_argument("-th", "--text_header", default='abstract', help="the column name for the free text")
    parser.add_argument("-dh", "--date_header", default=None, help="the column name for the date")

    # Word filters
    parser.add_argument("-fc", "--filter_columns", default=None,
                        help="list of columns with binary entries by which to filter the rows")
    parser.add_argument("-fb", "--filter_by", default='union', choices=['union', 'intersection'],
                        help="Returns filter: intersection where all are 'Yes' or '1'"
                             "or union where any are 'Yes' or '1' in the defined --filter_columns")
    parser.add_argument("-st", "--search_terms", type=str, nargs='+', default=[],
                        help="Search terms filter: search terms to restrict the tfidf dictionary. "
                             "Outputs will be related to search terms")

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

    # Normalize tf-idf scores by document length
    parser.add_argument("-ndl", "--normalize_doc_length", default=False, action="store_true",
                        help="normalize tf-idf scores by document length")

    # Time weighting
    parser.add_argument("-t", "--time", default=False, action="store_true", help="weight terms by time")

    # OUTPUT PARAMETERS
    # select outputs

    parser.add_argument("-o", "--output", nargs='*', default=[],
                        choices=['graph', 'wordcloud'],  # suppress table output option
                        help="Note that this can be defined multiple times to get more than one output. ")

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

    # emtech options
    parser.add_argument("-emt", "--emerging_technology", default=False, action="store_true",
                        help="denote whether emerging technology should be forecast")

    parser.add_argument("-pns", "--predictor_names", type=int, nargs='+', default=[2],
                        help=(", ".join([f"{index}. {value}" for index, value in enumerate(predictor_names)]))
                             + "; multiple inputs are allowed.\n")

    parser.add_argument("-nts", "--nterms", type=int, default=25,
                        help="number of terms to analyse")
    parser.add_argument("-mpq", "--minimum-per-quarter", type=int, default=15,
                        help="minimum number of patents per quarter referencing a term")
    parser.add_argument("-stp", "--steps_ahead", type=int, default=5,
                        help="number of steps ahead to analyse for")

    parser.add_argument("-cf", "--curve-fitting", default=False, action="store_true",
                        help="analyse using curve or not")

    parser.add_argument("-nrm", "--normalised", default=False, action="store_true",
                        help="analyse using normalised patents counts or not")

    args = parser.parse_args(command_line_arguments)

    args.path = 'data'
    return args


def main(supplied_args):
    paths = [os.path.join('outputs', 'reports'), os.path.join('outputs', 'wordclouds'),
             os.path.join('outputs', 'table'), os.path.join('outputs', 'emergence')]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    args = get_args(supplied_args)
    args_default = get_args([])
    argscheck = ArgsChecker(args, args_default)
    argscheck.checkargs()
    outputs = args.output[:]
    outputs.append('json_config')
    outputs.append('report')
    if args.term_counts:
        outputs.append('termcounts')

    docs_mask_dict = argscheck.get_docs_mask_dict()
    terms_mask_dict = argscheck.get_terms_mask_dict()

    doc_source_file_name = os.path.join(args.path, args.doc_source)

    if args.input_tfidf is None:
        pickled_tf_idf_path = None
    else:
        pickled_tf_idf_path = os.path.join('outputs', 'tfidf', args.input_tfidf)

    pipeline = Pipeline(doc_source_file_name, docs_mask_dict, pick_method=args.pick,
                        ngram_range=(args.min_ngrams, args.max_ngrams), normalize_rows=args.normalize_doc_length,
                        text_header=args.text_header, max_df=args.max_document_frequency,
                        term_counts=args.term_counts, user_ngrams=args.search_terms,
                        pickled_tf_idf_file_name=pickled_tf_idf_path,
                        output_name=args.outputs_name, emerging_technology=args.emerging_technology)

    pipeline.output(outputs, wordcloud_title=args.wordcloud_title, outname=args.outputs_name, nterms=50)

    # emtech integration
    if args.emerging_technology:
        from scripts.pipeline import PipelineEmtech

        if 0 in args.predictor_names:
            algs_codes = list(range(1, len(predictor_names)))
        else:
            algs_codes = args.predictor_names

        if isinstance(algs_codes, int):
            predictors_to_run = [predictor_names[algs_codes]]
        else:
            predictors_to_run = [predictor_names[i] for i in algs_codes]

        term_counts_data = pipeline.term_counts_data

        pipeline_emtech = PipelineEmtech(term_counts_data, m_steps_ahead=args.steps_ahead, curves=args.curve_fitting,
                                         nterms=args.nterms, minimum_patents_per_quarter=args.minimum_per_quarter,
                                         outname=args.outputs_name)

        for emergence in ['emergent', 'stationary', 'declining']:
            print(f'Running pipeline for "{emergence}"')

            if args.normalised:
                title = 'Forecasts Evaluation: Normalised Counts' if args.test else 'Forecasts: Normalised Counts'
            else:
                title = 'Forecasts Evaluation' if args.test else 'Forecasts'

            title += f' ({emergence})'

            html_results = pipeline_emtech.run(predictors_to_run, normalized=args.normalised, train_test=args.test,
                                        emergence=emergence)

            html_doc = f'''<!DOCTYPE html>
                <html lang="en">
                  <head>
                    <meta charset="utf-8">
                    <title>{title}</title>
                  </head>
                  <body>
                    <h1>{title}</h1>
                {html_results}
                  </body>
                </html>
                '''

            base_file_name = os.path.join('outputs', 'emergence', args.outputs_name + '_' + emergence)

            if args.normalised:
                base_file_name += '_normalised'

            if args.test:
                base_file_name += '_test'

            html_filename = base_file_name + '.html'

            with open(html_filename, 'w') as f:
                f.write(html_doc)

            print()


if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except PygramsException as err:
        print(f"pyGrams error: {err.message}")
