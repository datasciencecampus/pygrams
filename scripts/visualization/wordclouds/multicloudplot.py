import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from wordcloud import WordCloud


class MultiCloudPlot(object):

    def __init__(self, textsin=None, freqsin=None, max_words=200):
        string_to_float = {y: x for (x, y) in freqsin.items()}
        self.__wordcloud = None
        self.__texts = textsin
        self.__freqs = string_to_float
        self.__max_words = max_words

    def __generate_cloud(self):
        if self.__freqs is None:
            self.__wordcloud = WordCloud(width=1600, height=800, background_color="black", regexp=None,
                                         max_words=self.__max_words).generate(textin)
        else:
            self.__wordcloud = WordCloud(width=900, height=600, background_color="black",
                                         regexp=None, max_words=self.__max_words).generate_from_frequencies(
                self.__freqs)

    def plot_cloud(self, titlein, output_file_name, interpolation='bilinear'):

        fig1 = plt.figure(1)
        self.__generate_cloud()

        plt.imshow(self.__wordcloud, interpolation=interpolation)
        plt.grid(True)
        plt.axis("off")

        fig1.suptitle(titlein, fontsize=14)
        fig1.savefig(output_file_name, dpi=300)
        plt.show()
        plt.close(fig1)
