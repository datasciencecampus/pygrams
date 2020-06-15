import argparse
import csv
import os
import sys
import time


from scripts.pipeline import Pipeline
from scripts.utils.argschecker import ArgsChecker
from scripts.utils.pygrams_exception import PygramsException

predictor_names = ['All standard predictors', 'Naive', 'Linear', 'Quadratic', 'Cubic', 'ARIMA', 'Holt-Winters', 'SSM']


def get_args(command_line_arguments):
    parser = argparse.ArgumentParser(description="extract popular n-grams (words or short phrases)"
                                                 " from a corpus of documents",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
                                     conflict_handler='resolve')  # allows overridng of arguments

    # suppressed:________________________________________
    parser.add_argument("-ih", "--id_header", default=None, help=argparse.SUPPRESS)
    parser.add_argument("-c", "--cite", default=False, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-pt", "--path", default='data', help=argparse.SUPPRESS)
    parser.add_argument("-nmf", "--n_nmf_topics", type=int, default=0, help=argparse.SUPPRESS)
    # help="NMF topic modelling - number of topics (e.g. 20 or 40)")

    # Focus source and function
    parser.add_argument("-f", "--focus", default=None, choices=['set', 'chi2', 'mutual'],
                        help=argparse.SUPPRESS)
    parser.add_argument("-fs", "--focus_source", default='USPTO-random-1000.pkl.bz2', help=argparse.SUPPRESS)
    parser.add_argument("-tn", "--table_name", default=os.path.join('outputs', 'table', 'table.xlsx'),
                        help=argparse.SUPPRESS)

    parser.add_argument("-j", "--json", default=True, action="store_true",
                        help=argparse.SUPPRESS)
    # tf-idf score mechanics
    parser.add_argument("-p", "--pick", default='sum', choices=['median', 'max', 'sum', 'avg'],
                        help=argparse.SUPPRESS)
    parser.add_argument("-tst", "--test", default=False, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("-fb", "--filter_by", default='intersection', choices=['union', 'intersection'],
                        help=argparse.SUPPRESS)
    # end __________________________________________________

    # Input files
    parser.add_argument("-ds", "--doc_source", default='USPTO-random-1000.pkl.bz2',
                        help="the document source to process")
    parser.add_argument("-uc", "--use_cache", default=None,
                        help="Cache file to use, to speed up queries")

    # Document column header names
    parser.add_argument("-th", "--text_header", default='abstract', help="the column name for the free text")
    parser.add_argument("-dh", "--date_header", default=None, help="the column name for the date")

    # Word filters
    parser.add_argument("-fc", "--filter_columns", default=None,
                        help="list of columns with binary entries by which to filter the rows")

    parser.add_argument("-st", "--search_terms", type=str, nargs='+', default=[],
                        help="Search terms filter: search terms to restrict the tfidf dictionary. "
                             "Outputs will be related to search terms")
    parser.add_argument("-stthresh", "--search_terms_threshold", type=float,  default=0.75,
                        help="Provides the threshold of how related you want search terms to be "
                             "Values between 0 and 1: 0.8 is considered high")
    # Time filters
    parser.add_argument("-df", "--date_from", default=None,
                        help="The first date for the document cohort in YYYY/MM/DD format")
    parser.add_argument("-dt", "--date_to", default=None,
                        help="The last date for the document cohort in YYYY/MM/DD format")

    parser.add_argument("-tsdf", "--timeseries-date-from", default=None,
                        help="The first date for the document cohort in YYYY/MM/DD format")
    parser.add_argument("-tsdt", "--timeseries-date-to", default=None,
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

    # Remove noise terms before further processing
    parser.add_argument("-pt", "--prefilter_terms", type=int, default=100000,
                        help="Initially remove all but the top N terms by TFIDF score before pickling initial TFIDF"
                             " (removes 'noise' terms before main processing pipeline starts)")

    # OUTPUT PARAMETERS
    # select outputs

    parser.add_argument("-o", "--output", nargs='*', default=[],
                        choices=['wordcloud', 'multiplot'],  # suppress table output option
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

    # PATENT SPECIFIC SUPPORT
    parser.add_argument("-cpc", "--cpc_classification", default=None,
                        help="the desired cpc classification (for patents only)")

    # emtech options
    parser.add_argument("-ts", "--timeseries", default=False, action="store_true",
                        help="denote whether timeseries analysis should take place")

    parser.add_argument("-pns", "--predictor_names", type=int, nargs='+', default=[2],
                        help=(", ".join([f"{index}. {value}" for index, value in enumerate(predictor_names)]))
                             + "; multiple inputs are allowed.\n")

    parser.add_argument("-nts", "--nterms", type=int, default=25,
                        help="number of terms to analyse")
    parser.add_argument("-mpq", "--minimum-per-quarter", type=int, default=15,
                        help="minimum number of patents per quarter referencing a term")
    parser.add_argument("-stp", "--steps_ahead", type=int, default=5,
                        help="number of steps ahead to analyse for")

    parser.add_argument("-ei", "--emergence-index", default='porter', choices=('porter', 'net-growth'),
                        help="Emergence calculation to use")
    parser.add_argument("-sma", "--smoothing-alg", default='savgol', choices=('kalman', 'savgol'),
                        help="Time series smoothing to use")

    parser.add_argument("-exp", "--exponential_fitting", default=False, action="store_true",
                        help="analyse using exponential type fit or not")

    parser.add_argument("-nrm", "--normalised", default=False, action="store_true",
                        help="analyse using normalised patents counts or not")

    args = parser.parse_args(command_line_arguments)

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
    outputs.append('reports')
    outputs.append('json_config')
    if args.timeseries:
        outputs.append('timeseries')
    if args.n_nmf_topics > 0:
        outputs.append('nmf')

    docs_mask_dict = argscheck.get_docs_mask_dict()
    terms_mask_dict = argscheck.get_terms_mask_dict()

    doc_source_file_name = os.path.join(args.path, args.doc_source)

    pipeline = Pipeline(doc_source_file_name, docs_mask_dict, pick_method=args.pick,
                        ngram_range=(args.min_ngrams, args.max_ngrams), text_header=args.text_header,
                        cached_folder_name=args.use_cache,
                        max_df=args.max_document_frequency, user_ngrams=args.search_terms,
                        prefilter_terms=args.prefilter_terms, terms_threshold=args.search_terms_threshold,
                        output_name=args.outputs_name, calculate_timeseries=args.timeseries, m_steps_ahead=args.steps_ahead,
                        emergence_index=args.emergence_index, exponential=args.exponential_fitting, nterms=args.nterms,
                        patents_per_quarter_threshold=args.minimum_per_quarter, sma = args.smoothing_alg
                        )

    pipeline.output(outputs, wordcloud_title=args.wordcloud_title, outname=args.outputs_name,
                    nterms=args.num_ngrams_report, n_nmf_topics=args.n_nmf_topics)

    outputs_name = pipeline.outputs_folder_name

    # emtech integration
    if args.timeseries:
        if 0 in args.predictor_names:
            algs_codes = list(range(1, 7))
        else:
            algs_codes = args.predictor_names

        if isinstance(algs_codes, int):
            predictors_to_run = [predictor_names[algs_codes]]
        else:
            predictors_to_run = [predictor_names[i] for i in algs_codes]

        dir_path = os.path.join(outputs_name, 'emergence')
        os.makedirs(dir_path, exist_ok=True)

        for emergence in ['emergent', 'declining']:
            print(f'Running pipeline for "{emergence}"')

            if args.normalised:
                title = 'Forecasts Evaluation: Normalised Counts' if args.test else 'Forecasts: Normalised Counts'
            else:
                title = 'Forecasts Evaluation' if args.test else 'Forecasts'

            title += f' ({emergence})'

            html_results, training_values = pipeline.run(predictors_to_run, normalized=args.normalised,
                                                         train_test=args.test, emergence=emergence)
            if training_values is not None:
                # save training_values to csv file
                #
                # training_values:                                  csv file:
                # {'term1': [0,2,4,6], 'term2': [2,4,1,3]}          'term1', 0, 2, 4, 6
                #                                                   'term2', 2, 4, 1, 3
                #

                filename = os.path.join(dir_path,
                                        args.outputs_name + '_' + emergence + '_time_series.csv')
                with open(filename, 'w') as f:
                    w = csv.writer(f)
                    for key, values in training_values:
                        my_list = ["'" + str(key) + "'"] + values
                        w.writerow(my_list)

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

            base_file_name = os.path.join(dir_path, args.outputs_name + '_' + emergence)

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
        start = time.time()
        main(sys.argv[1:])
        end = time.time()
        diff = int(end - start)
        hours = diff // 3600
        minutes = diff // 60
        seconds = diff % 60

        print('')
        print(f"pyGrams query took {hours}:{minutes:02d}:{seconds:02d} to complete")
    except PygramsException as err:
        print(f"pyGrams error: {err.message}")
