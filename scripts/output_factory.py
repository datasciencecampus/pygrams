import os

from scripts.terms_graph import TermsGraph
from scripts.visualization.wordclouds.multicloudplot import MultiCloudPlot


def get(output_type, output, wordcloud_title=None, wordcloud_name=None, tfidf_reduce_obj=None, name=None, nterms=50):
    if output_type == 'report':
        full_path = os.path.join('outputs', 'reports', name)
        with open(full_path, 'w') as file:
            counter = 1
            for score, term in output:
                if counter<nterms:
                    file.write(f' {term:30} {score:f}\n')
                    print(f'{counter}. {term:30} {score:f}')
                    counter+=1
        print(output_type)
    elif output_type == 'fdg':
        print(output_type)
        dict_freqs = dict([(p[0], (p[1])) for p in output])
        dict_freqs_list = list(dict_freqs.items())[:nterms]
        graph = TermsGraph(dict_freqs_list, tfidf_reduce_obj)
        graph.save_graph_report(name, nterms)
        graph.save_graph("key-terms", 'data')

    elif output_type == 'wordcloud':
        print(output_type)
        dict_freqs = dict([(p[0], (p[1])) for p in output])
        wordcloud = MultiCloudPlot( freqsin=dict_freqs, max_words=len(output))
        wordcloud.plot_cloud(wordcloud_title, wordcloud_name)
    elif output_type == 'term_counts_mat':
        print(output_type)
    else:
        assert 0, "Bad output type: " + output_type