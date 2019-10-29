from os import path, makedirs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_multiplot(timeseries_terms_smooth, timeseries, test_terms, term_ngrams, lims, method='Net Growth',
                  category='emergent', output_name='multiplot'):
    # libraries and data
    import matplotlib.pyplot as plt
    import pandas as pd

    series_dict = {'x': range(len(timeseries[0]))}
    series_dict_smooth = {'x': range(len(timeseries_terms_smooth[0]))}

    for test_term in test_terms:
        term_index = term_ngrams.index(test_term)
        series_dict[term_ngrams[term_index]] = timeseries[term_index]
        series_dict_smooth[term_ngrams[term_index]] = timeseries_terms_smooth[term_index]


    # make a data frame
    df = pd.DataFrame(series_dict)
    df_smooth = pd.DataFrame(series_dict_smooth)

    # initialize the figure
    plt.style.use('seaborn-darkgrid')

    # create a color palette

    # multiple line plot
    fig, axs = plt.subplots(6, 5, figsize=(12, 10))
    num = 0
    for column in df.drop('x', axis=1):
        num += 1

        if num > len(test_terms):
            break

        # find the right spot on the plot
        current_graph = axs[(num - 1) % 6, (num - 1) // 6]

        # plot the lineplot
        current_graph.plot(df['x'], df[column], color='b', marker='', linewidth=1.4, alpha=0.9, label=column)
        current_graph.plot(df['x'], df_smooth[column], color='g', linestyle='-', marker='',
                           label='smoothed ground truth')

        current_graph.axvline(x=lims[0], color='k', linestyle='--')
        current_graph.axvline(x=lims[1], color='k', linestyle='--')

        # same limits for everybody!
        current_graph.set_xlim((0, max(series_dict['x'])))

        # not ticks everywhere
        if num in range(26):
            current_graph.tick_params(labelbottom='off')

        current_graph.tick_params(labelsize=8)

        # plt.tick_params(labelleft='off')

        # add title
        current_graph.title.set_text(column)
        current_graph.title.set_fontsize(10)
        current_graph.title.set_fontweight(0)

    # general title
    fig.suptitle(category + " keywords selection using the " + method + " index", fontsize=16, fontweight=0,
                 color='black', style='italic')

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # axis title
    plt.savefig(path.join('outputs', 'emergence', f'{output_name}-{category}-{method}.pdf'), dpi=300)
    plt.savefig(path.join('outputs', 'emergence', f'{output_name}-{category}-{method}.png'), dpi=300)


def get_counts_plot( timeseries_terms_smooth, terms_list, term_ngrams, dir_name, method='Net Growth',
                    category='emergent'):
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)
    terms = []
    escores = []
    counts = []
    for test_term in terms_list:
        terms.append(test_term[0])
        escores.append(test_term[1])
        term_index = term_ngrams.index(test_term[0])
        counts.append(sum(timeseries_terms_smooth[term_index]))

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.plot(counts, escores, 'bo', markersize=2)

    for i, txt in enumerate(terms):
        plt.text(counts[i], escores[i], txt, fontsize=8)
    # plt.xlim(1000, 10000)
    plt.suptitle(category + " keywords scatterplot score vs total counts for " + method + " index", fontsize=13,
                 fontweight=0,
                 color='black', style='italic')
    plt.xlabel('total counts')
    plt.ylabel('score')
    plt.savefig(path.join(dir_name, f'{category}_{method}.pdf'), dpi=300)


def boxplots(term_score_tups, dir_name, nterms):
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)

    tfidf = [x[0] for x in term_score_tups]

    ngrams=[]
    for x in term_score_tups:
        if len(x[1].split())==1:
            ngrams.append('uni')
        elif len(x[1].split())==2:
            ngrams.append('bi')
        elif len(x[1].split())==3:
            ngrams.append('tri')

    original = ['Yes' if x < nterms else 'No' for x in range(len(term_score_tups))]

    data = []
    for a, b, c in zip(tfidf, ngrams, original):
        data.append([a, b, c])

    ngram_df = pd.DataFrame(data, columns=['tfidf', 'ngram', 'subset'])

    sns.set(style="whitegrid")
    ax = sns.boxplot(x="ngram", y="tfidf", hue="subset", data = ngram_df, palette = "Set3", whis=9)
    ax.set_yscale('log', basey=2)
    ax.set_title(r'n-gram sum tfidf range')

    plt.savefig(path.join(dir_name, 'ngram_boxplots.png'))
    plt.close()


def plot_ngram_bars(ngrams1, ngrams2, dir_name):
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)
    labels = ['uni-grams', 'bi-grams', 'tri-grams']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, ngrams1, width, label='original')
    rects2 = ax.bar(x + width / 2, ngrams2, width, label='processed')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('counts')
    ax.set_title('ngram counts original vs processed tf-idf matrix')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig(path.join(dir_name, 'ngram_counts.png'))
    plt.close()


def tfidf_plot(tfidf_obj, message, dir_name=None):
    count_mat = tfidf_obj.count_matrix
    idf = tfidf_obj.idf
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)

    return tfidf_plot2(count_mat, idf, message, dir_name)


def tfidf_plot2(count_mat, idf, message, dir_name):
    counts_arr_sorted = count_mat.toarray().sum(axis=0)
    plt.scatter(counts_arr_sorted, idf, s=5)
    plt.xlabel('sum_tf')
    plt.ylabel('idf')
    plt.title(r'sum_tf vs idf | ' + message)
    plt.savefig(path.join(dir_name, '_'.join(message.split()) + '.png'))
    plt.close()


def scatter(arr1, arr2, xlab, ylab, title, dir_name):
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)
    plt.scatter(arr1, arr2, s=5)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(path.join(dir_name, '_'.join(title.split()) + '.png'))
    plt.close()


def escore_slope_plot(escore_slope_tup, dir_name, fname='escores_slopes', method='net-growth'):
    dir_name = dir_name.replace('cached', 'outputs', 1)
    if not path.isdir(dir_name):
        makedirs(dir_name)

    escores = [x[0] for x in escore_slope_tup]
    slopes = [x[1] for x in escore_slope_tup]

    plt.scatter(escores, slopes, s=5)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axvline(x=0, color='k', linestyle='--')

    # Add correlation line
    axes = plt.gca()
    m, b = np.polyfit(escores, slopes, 1)
    X_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
    plt.plot(X_plot, m * X_plot + b, '-')

    plt.xlabel('escores')
    plt.ylabel('slopes')
    plt.title(r'escores vs slopes 4 steps ahead | ' + method )
    plt.savefig(path.join(dir_name, fname+'.png'))
    plt.close()


def histogram(count_matrix):
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    counts_arr_sorted = count_matrix.toarray().sum(axis=0)
    num_bins = 100
    mu = np.mean(counts_arr_sorted)  # mean of distribution
    sigma = np.std(counts_arr_sorted)  # standard deviation of distribution
    # the histogram of the data
    n, bins, patches = plt.hist(counts_arr_sorted, num_bins, facecolor='blue', alpha=0.5)
    # add df vs tf plots here
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.yscale('log')
    plt.xlabel('Sum of term counts')
    plt.ylabel('Num Terms')
    plt.title(r'Histogram of sum of term frequencies')
    print(n)
    print(bins)
    print(patches)
    plt.ylim(bottom=1)
    plt.show()