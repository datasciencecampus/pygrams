import csv
import json
from bz2 import BZ2File
from os import makedirs, path
from pickle import dump

from scripts.nmf_wrapper import nmf_topic_modelling
from scripts.terms_graph import TermsGraph
from scripts.utils import utils
from scripts.utils.pygrams_exception import PygramsException
from scripts.visualization.wordclouds.multicloudplot import MultiCloudPlot


def dict_to_csv(timeseries_outputs, key, outputs_dir, method):
    dir_path_name = path.join(outputs_dir, 'emergent_'+key+'_'+method+'.csv')
    my_dict = timeseries_outputs[key]
    with open(dir_path_name, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for key, value in my_dict.items():
            writer.writerow([key, value])


def create(output_type, output, outputs_dir, emergence_list=[], wordcloud_title=None, tfidf_reduce_obj=None, name=None,
           nterms=50, timeseries_data=None, date_dict=None, pick=None, doc_pickle_file_name=None, nmf_topics=0,
           timeseries_outputs=None, method='net-growth'):
    dir_path = path.join(outputs_dir, output_type)
    makedirs(dir_path, exist_ok=True)

    if output_type == 'reports':
        filename_and_path = path.join(dir_path, name + '_keywords.txt')
        print()
        with open(filename_and_path, 'w') as file:
            counter = 0
            for score, term in output:
                if counter < nterms:
                    counter += 1
                    file.write(f' {term:30} {score:f}\n')
                    print(f'{counter}. {term:30} {score:f}')

    #TODO: Old code, remove or replace with chord diagram at some point
    elif output_type == 'graph':
        dict_freqs = dict([(p[0], (p[1])) for p in output])
        dict_freqs_list = list(dict_freqs.items())[:nterms]
        graph = TermsGraph(dict_freqs_list, tfidf_reduce_obj)
        name_and_path_report = path.join(outputs_dir,'reports', name + '_graph.txt')
        graph.save_graph_report(name_and_path_report, nterms)

        dir_visuals = path.join(outputs_dir, 'visuals')
        graph.save_graph(dir_visuals, "key-terms", 'data')

    elif output_type == 'wordcloud':
        dict_freqs = {p[0]: p[1] for p in output}
        wordcloud = MultiCloudPlot(freqsin=dict_freqs, max_words=len(output))
        filename_and_path = path.join(dir_path, name)
        wordcloud.plot_cloud(wordcloud_title, filename_and_path)

    elif output_type == 'timeseries':
        if timeseries_outputs is not None:
            dict_to_csv(timeseries_outputs, 'signal', dir_path, method)
            dict_to_csv(timeseries_outputs, 'signal_smooth', dir_path, method)
            dict_to_csv(timeseries_outputs, 'derivatives', dir_path, method)

            filename_and_path = path.join(dir_path, name + '_' + method + '_index.txt')
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

    elif output_type == 'json_config':
        # needs updating
        doc_pickle_file_name = path.abspath(doc_pickle_file_name)
        report_name_and_path = path.join(dir_path, name + '_keywords.txt')
        json_file_name = path.splitext(report_name_and_path)[0] + '_config.json'

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
                'pick': pick
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
        filename_and_path = path.join(dir_path, name + '_nmf.txt')
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

    else:
        raise PygramsException("Bad output type: " + output_type)
