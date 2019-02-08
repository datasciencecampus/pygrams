import json
from numba import jit
import os


class TermsGraph(object):
    MAX_NODES = 50
    MAX_LINKS = 10

    def __init__(self, list_tfidf_term, tfidf_obj):

        self.___ndocs = len(tfidf_obj.abstracts)
        self.__terms_list = [term for _, term in list_tfidf_term]
        self.__tfidf_term_list = list_tfidf_term
        self.__tfidf_obj = tfidf_obj
        self.__node_links_dict = self.__update_dict()
        self.__update_graph()

    @jit
    def __update_dict(self):
        node_links_dict = {}
        for term in self.__terms_list:
            node_links_dict[term]={}

        for idx in range(self.___ndocs):
            _, list_term_tfidf = self.__tfidf_obj.detect_popular_ngrams_in_docs_set(docs_set=[idx], number_of_ngrams_to_return=10)
            for idx_t1, term_tfidf_tup in enumerate(list_term_tfidf):
                if term_tfidf_tup[1] not in node_links_dict:
                        continue
                for idx_t2, term_freq2 in enumerate(list_term_tfidf):
                    if idx_t1 == idx_t2:
                        continue
                    weight = term_freq2[0] if term_freq2[1] not in self.__terms_list else 5*term_freq2[0]
                    if term_freq2[1] not in node_links_dict[term_tfidf_tup[1]]:
                        node_links_dict[term_tfidf_tup[1]][term_freq2[1]] = weight
                    else:
                        node_links_dict[term_tfidf_tup[1]][term_freq2[1]] += weight
        return node_links_dict

    @jit
    def __update_graph(self):
        nodes=[]
        links =[]
        node_min=float("inf")
        node_max=0

        link_min=float("inf")
        link_max=0
        for term_tup in self.__tfidf_term_list:
            node_min = node_min if term_tup[0] > node_min else term_tup[0]
            node_max = node_max if term_tup[0] < node_max else term_tup[0]
            for tup in self.__node_links_dict[term_tup[1]].items():
                link_min = node_min if tup[1] > link_min else tup[1]
                link_max = node_max if tup[1] < link_max else tup[1]

        for term_tup in self.__tfidf_term_list:
            term = term_tup[1]
            tfidf_score = term_tup[0]
            node = {'text': term, 'freq': self.normalize(tfidf_score, node_min, node_max)}
            nodes.append(node)
            d = sorted(self.__node_links_dict[term].items(), key=lambda x: x[1], reverse=True)

            for tup in d[:self.MAX_LINKS]:
                term_record = {'source': term, 'target': tup[0], 'size': self.normalize(tup[1], link_min, link_max)}
                links.append(term_record)
        self.__graph = {'nodes': nodes, 'links': links}

    def save_graph_report(self, args):

        links = self.__graph['links']

        with open(args.report_name[:len(args.report_name) - 4] + "_graph.txt", 'w') as file:
            counter = 1
            for score, term in self.__tfidf_term_list:
                file.write(f'{counter}. {term:10}:{score:1.2f}  -> ')
                print(f'{counter}. {term:10} -> ', end='', flush=True)
                counter += 1
                if counter > args.num_ngrams_report:
                    break
                out_str = []
                for link in links:
                    if term == link['source']:
                        target = link['target']
                        target_score = link['size']
                        out_str.append(f'{target:10}: {target_score:1.2f}')
                file.write(', '.join(out_str) + '\n')
                print(', '.join(out_str))

    def save_graph(self, fname, varname):

        graph = self.__graph
        links = graph['links']
        new_links = []
        for link in links:
            if link['target'] in self.__terms_list[:self.MAX_NODES] and link['source'] in self.__terms_list[:self.MAX_NODES]:
                new_links.append(link)

        graph['links'] = new_links
        graph['nodes'] = graph['nodes'][:self.MAX_NODES]
        file_name = os.path.join('outputs', 'visuals', fname + '.js')
        with open(file_name, 'w') as js_temp:
            js_temp.write(varname + " = '[")
            json.dump(graph, js_temp)
            js_temp.write("]'")
        file_name_jason = os.path.join('outputs', 'reports', fname + '.json')
        with open(file_name_jason, 'w') as js_temp:
            json.dump(graph, js_temp)

    @property
    def graph(self):
        return self.__graph

    def normalize(self,x, minx, maxx, offset=0.0):
        return((x-minx)/(maxx-minx)) + offset