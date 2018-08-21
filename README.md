[![build status](http://img.shields.io/travis/datasciencecampus/patent_app_detect/master.svg?style=flat)](https://travis-ci.org/datasciencecampus/patent_app_detect)
[![codecov](https://codecov.io/gh/datasciencecampus/patent_app_detect/branch/master/graph/badge.svg)](https://codecov.io/gh/datasciencecampus/patent_app_detect)
[![LICENSE.](https://img.shields.io/badge/license-OGL--3-blue.svg?style=flat)](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

# patent_app_detect 
 
## Description of tool

The deTecT tool is designed to derive popular terminology included within a particular patent technology area (CPC classification), based on text analysis of patent abstract information.  If the tool is targeted at the Y02 classification, for example, identified terms could include 'fuel cell' and 'heat exchanger'. A number of options are provided, for example to provide report, word cloud or graphical output. Some example outputs are shown below:

### Report

The score here is derived from the term tfidf values.

|Term			            |	     Score  |
| :------------------------ | ------------: |
|1. fuel cell               |       2.143778|
|2. heat exchanger          |       1.697166|
|3. exhaust gas             |       1.496812|
|4. combustion engine       |       1.480615|
|5. combustion chamber      |       1.390726|
|6. energy storage          |       1.302651|
|7. internal combustion     |       1.108040|
|8. positive electrode      |       1.100686|
|9. carbon dioxide          |       1.092638|
|10. control unit           |       1.069478|

### Word cloud

[Wordcloud example](https://github.com/datasciencecampus/detect/output/wordclouds/wordcloud_tech.png)

### Force directed graph

This output provides an interactive graph that shows connections between terms that are generally found in the same patent documents.

[fdg example](https://github.com/datasciencecampus/detect/fdg/index.html)

## How to install
### Windows ###
1. Please make sure Python 3.6 is installed and set at your path.  
   It can be installed from [this location](https://www.python.org/downloads/release/python-360/) selecting the *Windows x86 executable installer* option. When prompted, please check the box to set the paths and environment variables for you and you should be ready to go.

   ```
   python --version
   ```
   will show which python version is the default for your system.  

2. From the root directory (app_detect), run:
   ``` 
   setapp.bat
   ```
   This will install all the libraries and run some tests. If the tests pass, the app is ready to run.
   If any of the tests fail, please email thanasis.anthopoulos@ons.gov.uk or ian.grimstead@ons.gov.uk
   with a screenshot of the failure and we will get back to you.

## How to use

The program is command line driven, and called in the following manner:

```
python detect.py
```

The above produces a default report output of top ranked terms, using default parameters. Additional command line arguments provide alternative options, for example a word cloud or force directed graph (fdg) output. The option 'all', produces all three:

```
python detect.py -o='wordcloud'
python detect.py -o='fdg'
python detect.py -o='all'
```

### Choose patent source

This selects the set of patents for use during analysis. The default source is a random 1000 patent set from the USPTO, USPTO-random-1000. Datasets of 100, 1000, 10000, 100000, and 500000 patents are available.

```
python detect.py -ps=USPTO-random-10000
```

### Choose CPC classification

This subsets the chosen patents dataset to a particular CPC class, in this example Y02. In this case a larger patent source is generally required to allow for the reduction in patent numbers after subsetting.

```
python detect.py -cpc=Y02 -ps=USPTO-random-10000
```

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

This option utilises a second random patent dataset, by default USPTO-random-10000, whose terms are discounted from the chosen CPC classification to try and 'focus' the identified terms away from terms found more generally in the patent dataset. An example of choosing a larger 

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
