import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class MultiCloudPlot(object):

    def __init__(self, textsin, freqsin=None, max_words=200):
        self.__wordcloud = None
        self.__texts = textsin
        self.__freqs = freqsin
        self.__max_words = max_words

    def __generate_cloud(self, textin):
        if self.__freqs is None:
            self.__wordcloud = WordCloud(width=1600, height=800, background_color="black", regexp=None,
                                         max_words=self.__max_words).generate(textin)
        else:
            self.__wordcloud = WordCloud(width=900, height=600, background_color="black",
                                         regexp=None, max_words=self.__max_words).generate_from_frequencies(
                self.__freqs)

    def red_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(10, 100%%, %d%%)" % random.randint(40, 100)

    def amber_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(35, 100%%, %d%%)" % random.randint(40, 100)

    def green_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        return "hsl(150, 100%%, %d%%)" % random.randint(40, 100)

    def plot_cloud(self, titlein, output_file_name, interpolation='bilinear'):

        fig1 = plt.figure(1)
        self.__generate_cloud(self.__texts)

        plt.imshow(self.__wordcloud, interpolation=interpolation)
        plt.grid(True)
        plt.axis("off")

        fig1.suptitle(titlein, fontsize=14)
        plt.show()
        fig1.savefig(output_file_name, dpi=300)
        plt.close(fig1)
