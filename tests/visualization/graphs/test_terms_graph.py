import pandas as pd
import unittest

from scripts import FilePaths
from scripts.algorithms.tfidf import TFIDF, StemTokenizer
from scripts.visualization.graphs.terms_graph import TermsGraph


class TestGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_ngrams = 50
        df = pd.read_pickle(FilePaths.us_patents_random_1000_pickle_name)
        tfidf_obj = TFIDF(df, ngram_range=(2, 4), tokenizer=StemTokenizer(), normalize_doc_length=False)
        terms, freqs_list = tfidf_obj.detect_popular_ngrams_in_docs_set( number_of_ngrams_to_return = num_ngrams)
        graph_obj = TermsGraph(freqs_list, tfidf_obj)
        graph = graph_obj.graph
        cls.__links = graph['links']
        cls.__nodes = graph['nodes']

    def test_num_nodes(self):
        self.assertEquals(50, len(self.__nodes))

    def test_num_links(self):
        self.assertEquals(471, len(self.__links))

    def test_terms_in_nodes(self):
        self.assertIn({'text': 'central portion', 'freq': 0.054940822918518054}, self.__nodes)
        self.assertIn({'text': 'fluid commun', 'freq': 0.032458775009343}, self.__nodes)
        self.assertIn({'text': 'provid seed', 'freq': 0.014660739181626243}, self.__nodes)
        self.assertIn({'text': 'gate line', 'freq': 0.07741770854955801} , self.__nodes)

    def test_terms_in_links(self):
        self.assertIn({'source': 'semiconductor substrat', 'target': 'diffus barrier materi', 'size': 0.179044009721727}, self.__links)
        self.assertIn({'source': 'lock arm', 'target': 'swing arm', 'size': 0.0969491167354179}, self.__links)
        self.assertIn({'source': 'bodi portion', 'target': 'elastomer portion', 'size': 0.21718768050108705}, self.__links)
        self.assertIn({'source': 'central portion', 'target': 'convex outer surfac', 'size': 0.11142954674438521} , self.__links)


