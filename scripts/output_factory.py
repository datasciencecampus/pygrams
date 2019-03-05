import bz2
import os
import pickle

from scripts.terms_graph import TermsGraph
from scripts.visualization.wordclouds.multicloudplot import MultiCloudPlot


def create(output_type, output, wordcloud_title=None, tfidf_reduce_obj=None, name=None, nterms=50,
           term_counts_data=None, tfidf_obj=None):
    if output_type == 'report':
        filename_and_path = os.path.join('outputs', 'reports', name + '.txt')
        with open(filename_and_path, 'w') as file:
            counter = 1
            for score, term in output:
                if counter < nterms:
                    file.write(f' {term:30} {score:f}\n')
                    print(f'{counter}. {term:30} {score:f}')
                    counter += 1

    elif output_type == 'graph':
        dict_freqs = dict([(p[0], (p[1])) for p in output])
        dict_freqs_list = list(dict_freqs.items())[:nterms]
        graph = TermsGraph(dict_freqs_list, tfidf_reduce_obj)
        name_and_path = os.path.join('outputs', 'reports', name + '_graph.txt')
        graph.save_graph_report(name_and_path, nterms)
        graph.save_graph("key-terms", 'data')

    elif output_type == 'wordcloud':
        dict_freqs = dict([(p[0], (p[1])) for p in output])
        wordcloud = MultiCloudPlot(freqsin=dict_freqs, max_words=len(output))
        filename_and_path = os.path.join('outputs', 'wordclouds', name)
        wordcloud.plot_cloud(wordcloud_title, filename_and_path)

    elif output_type == 'termcounts':
        term_counts_filename = os.path.join('outputs', 'termcounts', name + '-term_counts.pkl.bz2')
        os.makedirs(os.path.dirname(term_counts_filename), exist_ok=True)
        with bz2.BZ2File(term_counts_filename, 'wb') as pickle_file:
            pickle.dump(term_counts_data, pickle_file, protocol=4)

    elif output_type == 'tfidf':
        tfidf_filename = os.path.join('outputs', 'tfidf', name + '-tfidf.pkl.bz2')
        os.makedirs(os.path.dirname(tfidf_filename), exist_ok=True)
        with bz2.BZ2File(tfidf_filename, 'wb') as pickle_file:
            pickle.dump(tfidf_obj, pickle_file, protocol=4)

    else:
        assert 0, "Bad output type: " + output_type
