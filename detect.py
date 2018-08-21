import argparse
import os

from pandas import Timestamp, read_pickle

from scripts import FilePaths
from scripts.algorithms.tfidf import LemmaTokenizer, TFIDF
from scripts.utils.pickle2df import PatentsPickle2DataFrame
from scripts.visualization.graphs.fdgprep import FDGPrep
from scripts.visualization.wordclouds.multicloudplot import MultiCloudPlot


def year2pandas_latest_date(year_in):
    if year_in == 0:
        return Timestamp.now()

    year_string = str(year_in) + '-12-31'
    return Timestamp(year_string)


def year2pandas_earliest_date(year_in):
    year_string = str(year_in) + '-01-01'
    return Timestamp(year_string)


def get_args():
    parser = argparse.ArgumentParser(description="create report, wordcloud, and fdg graph for patent texts")

    parser.add_argument("-f", "--focus", default=False, action="store_true",
                        help="clean output from terms that appear in general")
    parser.add_argument("-c", "--cite", default=False, action="store_true", help="weight terms by citations")
    parser.add_argument("-t", "--time", default=False, action="store_true", help="weight terms by time")

    parser.add_argument("-p", "--pick", default='sum', choices=['median', 'max', 'sum', 'avg'],
                        help="options are <median> <max> <sum> <avg>  defaults to sum. Average is over non zero values")
    parser.add_argument("-o", "--output", default='report', choices=['fdg', 'wordcloud', 'report', 'all'],
                        help="options are: <fdg> <wordcloud> <report> <all>")

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

    parser.add_argument("-rn", "--report_name", default=os.path.join('outputs', 'reports', 'report_tech.txt'),
                        help="report filename")
    parser.add_argument("-wn", "--wordcloud_name", default=os.path.join('outputs', 'wordclouds', 'wordcloud-tech.png'),
                        help="wordcloud filename")
    parser.add_argument("-wt", "--wordcloud_title", default='tech terms', help="wordcloud title")

    parser.add_argument("-cpc", "--cpc_classification", default=None, help="the desired cpc classification")

    args = parser.parse_args()
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


def get_tfidf(args, filename, cpc):
    date_from = year2pandas_earliest_date(args.year_from)
    date_to = year2pandas_latest_date(args.year_to)

    df = PatentsPickle2DataFrame(filename, classification=cpc, date_from=date_from, date_to=date_to).data_frame
    check_cpc_between_years(args, df)
    return TFIDF(df, tokenizer=LemmaTokenizer(), ngram_range=(args.min_n, args.max_n))


def main():
    paths = [os.path.join('outputs', 'reports'), os.path.join('outputs', 'json'), os.path.join('outputs', 'wordclouds')]
    for path in paths:
        os.makedirs(path, exist_ok=True)

    args = get_args()
    checkargs(args)
    path = os.path.join('data', args.patent_source + ".pkl.bz2")
    tfidf = get_tfidf(args, path, args.cpc_classification)

    newtfidf = None
    if args.focus:
        path2 = os.path.join('data', args.focus_source + ".pkl.bz2")
        newtfidf = get_tfidf(args, path2, None)

    out = args.output

    if out == 'report':
        run_report(args, tfidf, newtfidf)
    elif out == 'wordcloud' or out == 'all':
        run_report(args, tfidf, newtfidf, wordclouds=True)

    if out == 'fdg' or out == 'all':
        run_fdg(args, tfidf, newtfidf)


def run_report(args, tfidf, tfidf_random=None, wordclouds=False):
    num_ngrams = max(args.num_ngrams_report, args.num_ngrams_wordcloud)

    citation_count_dict = None
    if args.cite:
        citation_count_dict = read_pickle(FilePaths.us_patents_citation_dictionary_1of2_pickle_name)
        citation_count_dict_pt2 = read_pickle(FilePaths.us_patents_citation_dictionary_2of2_pickle_name)
        citation_count_dict.update(citation_count_dict_pt2)

    terms, ngrams_scores_tuple = tfidf.detect_popular_ngrams_in_corpus(number_of_ngrams_to_return=4 * num_ngrams,
                                                                       pick=args.pick, time=args.time,
                                                                       citation_count_dict=citation_count_dict)
    set_terms = set(terms) if not args.focus \
        else tfidf.detect_popular_ngrams_in_corpus_excluding_common(tfidf_random,
                                                                    number_of_ngrams_to_return=4 * num_ngrams,
                                                                    pick=args.pick, time=args.time,
                                                                    citation_count_dict=citation_count_dict)

    dict_freqs = dict([((p[1]), p[0]) for p in ngrams_scores_tuple if p[1] in set_terms])

    with open(args.report_name, 'w') as file:
        counter = 1
        for term, score in dict_freqs.items():

            line = ' {:30} {:f}\n'.format(term, score)
            file.write(line)
            print(str(counter) + "." + line)
            counter += 1
            if counter > args.num_ngrams_report:
                break

    if wordclouds:
        doc_all = ' '.join(set_terms)
        wordcloud = MultiCloudPlot(doc_all, freqsin=dict_freqs, max_words=args.num_ngrams_wordcloud)
        wordcloud.plot_cloud(args.wordcloud_title, args.wordcloud_name)


def run_fdg(args, tf_idf, tf_idf2=None):
    graph = FDGPrep(args.num_ngrams_fdg)
    graph.fdg_tfidf(tf_idf, tf_idf2, args)
    graph.save_graph("key-terms", 'data')


if __name__ == '__main__':
    main()
