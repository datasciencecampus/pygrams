[![build status](http://img.shields.io/travis/datasciencecampus/patent_app_detect/master.svg?style=flat)](https://travis-ci.org/datasciencecampus/patent_app_detect)
[![codecov](https://codecov.io/gh/datasciencecampus/patent_app_detect/branch/master/graph/badge.svg)](https://codecov.io/gh/datasciencecampus/patent_app_detect)
[![LICENSE.](https://img.shields.io/badge/license-OGL--3-blue.svg?style=flat)](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

# patent_app_detect 
 
## Description of tool

The tool is designed to derive popular terminology included within a particular patent technology area ([CPC classification](https://www.epo.org/searching-for-patents/helpful-resources/first-time-here/classification/cpc.html)), based on text analysis of patent abstract information.  If the tool is targeted at the [Y02 classification](https://www.epo.org/news-issues/issues/classification/classification.html), for example, identified terms could include 'fuel cell' and 'heat exchanger'. A number of options are provided, for example to provide report, word cloud or graphical output. Some example outputs are shown below:

### Report

The score here is derived from the term [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) values using the Y02 classification on a 10,000 random sample of patents. The terms are all bigrams in this example.

|Term			            |	    TF-IDF Score  |
| :------------------------ | -------------------:|
|1. fuel cell               |       2.143778      |
|2. heat exchanger          |       1.697166      |
|3. exhaust gas             |       1.496812      |
|4. combustion engine       |       1.480615      |
|5. combustion chamber      |       1.390726      |
|6. energy storage          |       1.302651      |
|7. internal combustion     |       1.108040      |
|8. positive electrode      |       1.100686      |
|9. carbon dioxide          |       1.092638      |
|10. control unit           |       1.069478      |

### Word cloud

Here is a [wordcloud](https://raw.githubusercontent.com/datasciencecampus/patent_app_detect/master/outputs/wordclouds/wordcloud_tech.png) using the Y02 classification on a 10,000 random sample of patents. The greater the tf-idf score, the larger the font size of the term.

### Force directed graph

This output provides an interactive graph in the to be viewed in a web browser (you need to locally open the file ```outputs/fdg/index.html```). The graph shows connections between terms that are generally found in the same patent documents. The example wordcloud in the ```outputs/fdg``` folder was created using the Y02 classification on a 10,000 random sample of patents.

## How to install

The tool has been developed to work on both Windows and MacOS. To install:

1. Please make sure Python 3.6 is installed and set at your path.  
   It can be installed from [this location](https://www.python.org/downloads/release/python-360/) selecting the *relevant installer for your opearing system*. When prompted, please check the box to set the paths and environment variables for you and you should be ready to go. Python can also be installed as part of Anaconda [here](https://www.anaconda.com/download/#macos).

   To check the Python version default for your system, run the following in command line/terminal:

   ```
   python --version
   ```
   
   **_Note_**: If Python 2 is the default Python version, but you have installed Python 3.6, your path may be setup to use `python3` instead of `python`.
   
2. To install the packages and dependencies for the tool, from the root directory (patent_app_detect) run:
   ``` 
   pip install -e .
   ```
   This will install all the libraries and run some tests. If the tests pass, the app is ready to run. If any of the tests fail, please email thanasis.anthopoulos@ons.gov.uk or ian.grimstead@ons.gov.uk
   with a screenshot of the failure and we will get back to you.

## How to use

The program is command line driven, and called in the following manner:

```
python detect.py
```

The above produces a default report output of top ranked terms, using default parameters. Additional command line arguments provide alternative options, for example a word cloud or force directed graph (fdg) output. The option 'all', produces all three:

```
python detect.py -o='report' (using just `python detect.py` defaults to this option)
python detect.py -o='wordcloud'
python detect.py -o='fdg'
python detect.py -o='all'
```

### Choosing patent source

This selects the set of patents for use during analysis. The default source is a pre-created random 1,000 patent dataset from the USPTO, `USPTO-random-1000`. Pre-created datasets of 100, 1,000, 10,000, 100,000, and 500,000 patents are available in the `./data` folder. For example using:

```
python detect.py -ps=USPTO-random-10000
```

Will run the tool for a pre-created random dataset of 10,000 patents.

### Additional patent sources

Patent datasets are stored in the sub-folder ```data```, we have supplied the following files:
- ```USPTO-random-100.pkl.bz2```
- ```USPTO-random-1000.pkl.bz2```
- ```USPTO-random-10000.pkl.bz2```
- ```USPTO-random-100000.pkl.bz2```
- ```USPTO-random-500000.pkl.bz2```

The command ```python detect.py -ps=USPTO-random-10000``` instructs the program to load a pickled data frame of patents
from a file located in ```data/USPTO-random-10000.pkl.bz2```. Hence ```-ps=NAME``` looks for ```data/NAME.pkl.bz2```.

We have hosted larger datasets on a google drive, as the files are too large for GitHub version control. We have made available:
- All USPTO patents from 2004 (477Mb): [USPTO-all.pkl.bz2](https://drive.google.com/drive/folders/1d47pizWdKqtORS1zoBzsk3tLk6VZZA4N)
 
To use additional files, follow the link and download the pickle file into the data folder. Access the new data
with ```-ps=NameWithoutFileExtension```; for example, ```USPTO-all.pkl.bz2``` would be loaded with ```-ps=USPTO-all```.

Note that large datasets will require a large amount of system memory (such as 64Gb), otherwise it will process very slowly
as virtual memory (swap) is very likely to be used.

### Choosing CPC classification

This subsets the chosen patents dataset to a particular Cooperative Patent Classification (CPC) class, for example Y02. The Y02 classification is for "technologies or applications for mitigation or adaptation against climate change". In this case a larger patent dataset is generally required to allow for the reduction in patent numbers after subsetting. An example script is:

```
python detect.py -cpc=Y02 -ps=USPTO-random-10000
```

In the console the number of subset patents will be stated. For example, for `python detect.py -cpc=Y02 -ps=USPTO-random-10000` the number of Y02 patents is 197. Thus, the tf-idf will be run for 197 patents.


### Term n-gram limits

Terms identified may be unigrams, bigrams, or trigrams. The following arguments set the ngram limits for 2-3 word terms (which are the default values).
```
python detect.py -mn=2 -mx=3
```

### Time limits
This will restrict the patents cohort to only those from 2000 up to now.

```
python detect.py -yf=2000
```

This will restrict the patents cohort to only those between 2000 - 2016.

```
python detect.py -yf=2000 -yt=2016
```
### Time weighting

This option applies a linear weight that starts from 0.01 and ends at 1 between the time limits.
```
python detect.py -t
```

### Citation weighting

This will weight the term tfidf scores by the number of citations each patent has. The weight is a normalised value between 0 and 1 with the higher the number indicating a higher number of citations.

```
python detect.py -c
```

### Term focus

This option utilises a second random patent dataset, by default `USPTO-random-10000`, whose terms are discounted from the chosen CPC classification to try and 'focus' the identified terms away from terms found more generally in the patent dataset. An example of choosing a larger 

```
python detect.py -f
```

### Choose focus source

This selects the set of patents for use during the term focus option, for example for a larger dataset.

```
python detect.py -fs=USPTO-random-100000
```

### Config files

There are three configuration files available inside the config directory:

- stopwords_glob.txt
- stopwords_n.txt
- stopwords_uni.txt

The first file (stopwords_glob.txt) contains stopwords that are applied to all ngrams.
The second file contains stopwords that are applied to all n-grams for n>1 and the last file (stopwords_uni.txt) contain stopwords that apply only to unigrams. The users can append stopwords into this files, to stop undesirable output terms.

## Help

A help function details the range and usage of these command line arguments:
```
python detect.py -h
```

An edited version of the help output is included below. This starts with a summary of arguments:

```
python detect.py -h
usage: detect.py [-h] [-f] [-c] [-t] [-p {median,max,sum,avg}]
                 [-o {fdg,wordcloud,report,all}] [-yf YEAR_FROM] [-yt YEAR_TO]
                 [-np NUM_NGRAMS_REPORT] [-nd NUM_NGRAMS_WORDCLOUD]
                 [-nf NUM_NGRAMS_FDG] [-ps PATENT_SOURCE] [-fs FOCUS_SOURCE]
                 [-mn {1,2,3}] [-mx {1,2,3}] [-rn REPORT_NAME]
                 [-wn WORDCLOUD_NAME] [-wt WORDCLOUD_TITLE]
                 [-cpc CPC_CLASSIFICATION]

create report, wordcloud, and fdg graph for patent texts

```
It continues with a detailed description of the arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  -f, --focus           clean output from terms that appear in general
  -c, --cite            weight terms by citations
  -t, --time            weight terms by time
  -p {median,max,sum,avg}, --pick {median,max,sum,avg}
                        options are <median> <max> <sum> <avg> defaults to
                        sum. Average is over non zero values
  -o {fdg,wordcloud,report,all}, --output {fdg,wordcloud,report,all}
                        options are: <fdg> <wordcloud> <report> <all>
  -yf YEAR_FROM, --year_from YEAR_FROM
                        The first year for the patent cohort
  -yt YEAR_TO, --year_to YEAR_TO
                        The last year for the patent cohort (0 is now)
  -np NUM_NGRAMS_REPORT, --num_ngrams_report NUM_NGRAMS_REPORT
                        number of ngrams to return for report
  -nd NUM_NGRAMS_WORDCLOUD, --num_ngrams_wordcloud NUM_NGRAMS_WORDCLOUD
                        number of ngrams to return for wordcloud
  -nf NUM_NGRAMS_FDG, --num_ngrams_fdg NUM_NGRAMS_FDG
                        number of ngrams to return for fdg graph
  -ps PATENT_SOURCE, --patent_source PATENT_SOURCE
                        the patent source to process
  -fs FOCUS_SOURCE, --focus_source FOCUS_SOURCE
                        the patent source for the focus function
  -mn {1,2,3}, --min_n {1,2,3}
                        the minimum ngram value
  -mx {1,2,3}, --max_n {1,2,3}
                        the maximum ngram value
  -rn REPORT_NAME, --report_name REPORT_NAME
                        report filename
  -wn WORDCLOUD_NAME, --wordcloud_name WORDCLOUD_NAME
                        wordcloud filename
  -wt WORDCLOUD_TITLE, --wordcloud_title WORDCLOUD_TITLE
                        wordcloud title
  -cpc CPC_CLASSIFICATION, --cpc_classification CPC_CLASSIFICATION
                        the desired cpc classification

```

## Acknowledgements

### Patent data

Patent data was obtained from the [United States Patent and Trademark Office (USPTO)](https://www.uspto.gov) through the [Bulk Data Storage System (BDSS)](https://bulkdata.uspto.gov). In particular we used the `Patent Grant Full Text Data/APS (JAN 1976 - PRESENT)` dataset, using the data from 2004 onwards in XML 4.* format.
