import bz2
import json
import pickle
from os import makedirs, path

from scripts.nmf_wrapper import nmf_topic_modelling
from scripts.terms_graph import TermsGraph
from scripts.utils import utils
from scripts.visualization.wordclouds.multicloudplot import MultiCloudPlot


def create(output_type, output, emergence_list=[], wordcloud_title=None, tfidf_reduce_obj=None, name=None, nterms=50,
           timeseries_data=None, date_dict=None, pick=None, doc_pickle_file_name=None, time=None, nmf_topics=0):

    if output_type == 'report':
        filename_and_path = path.join('outputs', 'reports', name + '.txt')
        print()
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
        name_and_path = path.join('outputs', 'reports', name + '_graph.txt')
        graph.save_graph_report(name_and_path, nterms)
        graph.save_graph("key-terms", 'data')

    elif output_type == 'wordcloud':
        dict_freqs = dict([(p[0], (p[1])) for p in output])
        wordcloud = MultiCloudPlot(freqsin=dict_freqs, max_words=len(output))
        filename_and_path = path.join('outputs', 'wordclouds', name)
        wordcloud.plot_cloud(wordcloud_title, filename_and_path)

    elif output_type == 'termcounts':
        term_counts_filename = path.join('outputs', 'termcounts', name + '-term_counts.pkl.bz2')
        makedirs(path.dirname(term_counts_filename), exist_ok=True)
        with bz2.BZ2File(term_counts_filename, 'wb') as pickle_file:
            pickle.dump(timeseries_data, pickle_file, protocol=4)

    elif output_type == 'json_config':
        doc_pickle_file_name = path.abspath(doc_pickle_file_name)
        report_name_and_path = path.join('outputs', 'reports', name + '.txt')
        json_file_name = path.splitext(report_name_and_path)[0] + '.json'

        json_data = {
            'paths': {
                'data': doc_pickle_file_name,
                'tech_report': report_name_and_path
            },
            'month_year': {
                'from': None,
                'to': None
            },
            'parameters': {
                'pick': pick,
                'time': time
            }
        }

        if date_dict is not None:
            json_data['month_year']['from'] = date_dict['from']
            json_data['month_year']['to'] = date_dict['to']

        with open(json_file_name, 'w') as json_file:
            json.dump(json_data, json_file)
    elif output_type =='nmf':
        # topic modelling
        topic_terms_to_print = 10
        nmf = nmf_topic_modelling(nmf_topics, tfidf_reduce_obj.tfidf_masked)
        filename_and_path = path.join('outputs', 'reports', name + '_nmf.txt')
        with open(filename_and_path, 'w') as file:

            # print topics
            print()
            print('*** NMF topic modelling (experimental only) ***')
            file.write('*** NMF topic modelling (experimental only) *** \n')
            print('Topics:')
            file.write('Topics \n')
            feature_names = tfidf_reduce_obj.feature_names
            for topic_idx, term_weights in enumerate(nmf.components_):
                print("%d:" % (topic_idx), end='')
                file.write("%d: " % (topic_idx))
                topic_names = ", ".join(
                    [feature_names[i] for i in term_weights.argsort()[:-topic_terms_to_print - 1:-1]])
                print(topic_names)
                file.write(topic_names + '\n')
            print()
            file.write('\n')
    elif output_type == 'emergence_report':
        filename_and_path = path.join('outputs', 'reports', name + '_timeseries.csv')
        with open(filename_and_path, 'w') as file:
            print()
            print('Emergent')
            file.write('Emergent\n')
            for tup in emergence_list[:nterms]:
                print(tup[0] + ": " + str(tup[1]))
                file.write(tup[0] + ": " + str(tup[1]) + '\n')
            print()
            file.write('\n')

            print('Stationary')
            file.write('Stationary\n')
            stationary = utils.stationary_terms(emergence_list, nterms)
            for tup in stationary:
                print(tup[0] + ": " + str(tup[1]))
                file.write(tup[0] + ": " + str(tup[1]) + '\n')
            print()
            file.write('\n')

            print('Declining')
            file.write('Declining' + '\n')
            for tup in emergence_list[-nterms:]:
                print(tup[0] + ": " + str(tup[1]))
                file.write(tup[0] + ": " + str(tup[1]) + '\n')
            print()
            file.write('\n')
    else:
        assert 0, "Bad output type: " + output_type
