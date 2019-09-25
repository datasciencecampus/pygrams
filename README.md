[![build status](http://img.shields.io/travis/datasciencecampus/pyGrams/master.svg?style=flat)](https://travis-ci.org/datasciencecampus/pyGrams)
[![Build status](https://ci.appveyor.com/api/projects/status/oq49c4xuhd8j2mfp/branch/master?svg=true)](https://ci.appveyor.com/project/IanGrimstead/patent-app-detect/branch/master)
[![codecov](https://codecov.io/gh/datasciencecampus/pyGrams/branch/master/graph/badge.svg)](https://codecov.io/gh/datasciencecampus/pyGrams)
[![LICENSE.](https://img.shields.io/badge/license-OGL--3-blue.svg?style=flat)](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)



<p align="center"><img align="center" src="meta/images/pygrams-logo.png" width="400px"></p>

## Description of tool

This python-based app (`pygrams.py`) is designed to extract popular or emergent n-grams/terms (words or short phrases) from free text within a large (>1,000) corpus of documents. Example corpora of granted patent document abstracts are included for testing purposes.

The app pipeline (more details in the user option section):
1. **[Input Text Data](#input-text-data)** Text data can be input by several text document types (ie. csv, xls, pickled python dataframes, etc) 
2. **[TFIDF Dictionary](#tfidf-dictionary)**  This is the processed list of terms (ngrams) out of the whole corpus. These terms are the columns of the [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) sparse matrix. The user can control the following parameters: minimum document frequency, stopwords, ngram range. 
3. **TFIDF Computation** Grab a coffee if your text corpus is long (>1 million docs) :)
4. **Filters** These are filters to use on the computed TFIDF matrix. They consist of document filters and term filters
   1. **[Document Filters](#document-filters)** These filters work on document level. Examples are: date range, column features (eg. cpc classification).
   2. **[Term Filters](#term-filters)** These filters work on term level. Examples are: search terms list (eg. pharmacy, medicine, chemist)
5. **Mask the TFIDF Matrix** Apply the filters to the TFIDF matrix
6. **[Emergence](#emergence-calculations)**
   1. **[Emergence Calculations](#emergence-calculations)** Options include [Porter 2018](https://www.researchgate.net/publication/324777916_Emergence_scoring_to_identify_frontier_RD_topics_and_key_players) emergence calculations, curve fitting, or calculations designed to favour exponential like emergence. 
   2. **[Emergence Forecasts](#emergence-forecasts)** Options include ARIMA, linear and quadratic regression, Holt-Winters, LSTMs. 
8. **[Outputs](#outputs)** The default 'report' output is a ranked and scored list of 'popular' ngrams or emergent ones if selected. Other outputs include a 'graph summary', word cloud and an html document as emergence report.

## Installation guide

pyGrams.py has been developed to work on both Windows and MacOS. To install:

1. Please make sure Python 3.6 is installed and set in your path.  

   To check the Python version default for your system, run the following in command line/terminal:

   ```
   python --version
   ```

   **_Note_**: If Python 2.x is the default Python version, but you have installed Python 3.x, your path may be setup to use `python3` instead of `python`.

2. To install pyGrams packages and dependencies, from the root directory (./pyGrams) run:

   ``` 
   pip install -e .
   ```

   This will install all the libraries and then download their required datasets (namely NLTK's data). Once installed, 
   setup will run some tests. If the tests pass, the app is ready to run. If any of the tests fail, please email [ons.patent.explorer@gmail.com](mailto:ons.patent.explorer@gmail.com) with a screenshot of the failure so that we may get back to you, or alternatively open a [GitHub issue here](https://github.com/datasciencecampus/pyGrams/issues).

### System Performance

The system performance was tested using a 2.7GHz Intel Core i7 16GB MacBook Pro using 3.2M US patent abstracts from approximately 2005 to 2018. Indicatively, it initially takes about 6 hours to produce a specially optimised 100,000 term TFIDF Dictionary with a file size under 100MB. Once this is created however, it takes approximately 1 minute to run a pyGrams popular terminology query, or approximately 7 minutes for an emerging terminology query.

## User guide

pyGrams is command line driven, and called in the following manner:

```
python pygrams.py
```

### Input Text Data

#### Selecting the document source (-ds)

This argument is used to select the corpus of documents to analyse. The default source is a pre-created random 1,000 patent dataset from the USPTO, `USPTO-random-1000.pkl.bz2`. 

Pre-created datasets of 100, 1,000, 10,000, 100,000, and 500,000 patents are available in the `./data` folder:

- ```USPTO-random-100.pkl.bz2```
- ```USPTO-random-1000.pkl.bz2```
- ```USPTO-random-10000.pkl.bz2```
- ```USPTO-random-100000.pkl.bz2```
- ```USPTO-random-500000.pkl.bz2```

For example, to load the 10,000 pickled dataset for patents, use:

```
python pygrams.py -ds=USPTO-random-10000.pkl.bz2
```

To use your own document dataset, please place in the `./data` folder of pyGrams. File types currently supported are:

- pkl.bz2: compressed pickle file containing a dataset
- xlsx: new Microsoft excel format
- xls: old Microsoft excel format
- csv: comma separated value file (with headers)

Datasets should contain the following columns:

|Column			    |      Required?      |        Comments           |
|:---------------- | ------------------- | -------------------------:|
|Free text field   |       Yes           | Terms extracted from here |
|Date              |       Optional      | Compulsory for emergence  |
|Other headers     |       Optional      | Can filter by content     |

#### Selecting the column header names (-th, -dh)

When loading a document dataset, you will need to provide the column header names for each, using:

- `-th`: free text field column (default is 'text')
- `-dh`: date column (default is 'date', format is 'YYYY/MM/DD')

For example, for a corpus of book blurbs you could use:

```
python pygrams.py -th='blurb' -dh='published_date'
```

#### Using cached files to speed up processing (-uc)

In order save processing time, at various stages of the pipeline, we cache data structures that are costly and slow to compute, like the compressed tf-idf matrix, the timeseries matrix, the smooth series and its derivatives from kalman filter and others:

```
python pygrams.py -uc all-mdf-0.05-200501-201841
```

### TFIDF Dictionary

#### N-gram selection (-mn, -mx)

An n-gram is a contiguous sequence of n items ([source](https://en.wikipedia.org/wiki/N-gram)). N-grams can be unigrams (single words, e.g., vehicle), bigrams (sequences of two words, e.g., aerial vehicle), trigrams (sequences of three words, e.g., unmanned aerial vehicle) or any n number of continuous terms. 

The following arguments will set the n-gram limit to be, e.g. unigrams, bigrams, and trigrams (the default):

```
python pygrams.py -mn=1 -mx=3
```

To analyse only unigrams:

```
python pygrams.py -mn=1 -mx=1
```

#### Maximum document frequency (-mdf)

Terms identified are filtered by the maximum number of documents that use this term; the default is 0.05, representing an upper limit of 5% of documents containing this term. If a term occurs in more that 5% of documents it is rejected.

For example, to set the maximum document frequency to 5% (the default), use:

```
python pygrams.py -mdf 0.05
```

Using a small (5% or less) maximum document frequency may help remove generic words, or stop words.

#### Stopwords

There are three configuration files available inside the config directory:

- stopwords_glob.txt
- stopwords_n.txt
- stopwords_uni.txt

The first file (stopwords_glob.txt) contains stopwords that are applied to all n-grams. The second file contains stopwords that are applied to all n-grams for n > 1 (bigrams and trigrams). The last file (stopwords_uni.txt) contains stopwords that apply only to unigrams. The users can append stopwords into this files, to stop undesirable output terms.

#### Pre-filter Terms

Given that many of the terms will actually be very rare, they will not be of use when looking for popular terms.
The total number of terms can easily exceed 1,000,000 and slow down pygrams with
irrelevant terms. To circumvent this, a prefilter is applied as soon as the TFIDF matrix is created
which will retain the highest scoring terms by TFIDF (as is calculated and reported at the end of the main pipeline).
The default is to retain the top 100,000 terms; setting it to 0 will disable it, viz:
```
python pygrams.py -pt 0
```
Or changed to a different threshold such as 10,000 terms (using the longer argument name for comparison):
```
python pygrams.py -prefilter_terms 10000
```
Note that the prefilter will change TFIDF results as it will remove rare n-grams - which will
result in bi-grams & tri-grams having increased scores when rare uni-grams and bi-grams are removed, as we
unbias results to avoid double or triple counting contained n-grams.

### Document Filters

#### Time filters (-df, -dt)

This argument can be used to filter documents to a certain timeframe. For example, the below will restrict the document cohort to only those from 20 Feb 2000 up to now (the default start date being 1 Jan 1900).

```
python pygrams.py -dh publication_date -df=2000/02/20
```

The following will restrict the document cohort to only those between 1 March 2000 and 31 July 2016.

```
python pygrams.py -dh publication_date -df=2000/03/01 -dt=2016/07/31
```

#### Column features filters (-fh, -fb)

If you want to filter results, such as for female, British in the example below, you can specify the column names you wish to filter by, and the type of filter you want to apply, using:

- `-fh`: the list of column names (default is None)
- `-fb`: the type of filter (choices are `'union'` (default), where all fields need to be 'yes', or `'intersection'`, where any field can be 'yes') 

```
python pygrams.py -fh=['female','british'] -fb='union'
```

This filter assumes that values are '0'/'1', or 'Yes'/'No'.


#### Choosing CPC classification (Patent specific) (-cpc)

This subsets the chosen patents dataset to a particular Cooperative Patent Classification (CPC) class, for example Y02. The Y02 classification is for "technologies or applications for mitigation or adaptation against climate change". An example script is:

```
python pygrams.py -cpc=Y02 -ds=USPTO-random-10000.pkl.bz2
```

In the console the number of subset patents will be stated. For example, for `python pygrams.py -cpc=Y02 -ps=USPTO-random-10000.pkl.bz2` the number of Y02 patents is 197. Thus, the TFIDF will be run for 197 patents.

### Term Filters

#### Search terms filter (-st)

This subsets the TFIDF term dictionary by only keeping terms related to the given search terms.
```
python pygrams.py -st pharmacy medicine chemist
```

### Timeseries Calculations

#### Timeseries (-ts)

An option to choose between popular or emergent terminology outputs. Popular terminology is the default option; emergent terminology can be used by typing:

```
python pygrams.py -ts
```

#### Emergence Index (-ei)

An option to choose between quadratic fitting, [Porter 2018](https://www.researchgate.net/publication/324777916_Emergence_scoring_to_identify_frontier_RD_topics_and_key_players) or gradients from state-space model using kalman filter smoothing  emergence indexes. Porter is used by default; quadratic fitting can be used instead, for example:

```
python pygrams.py -ts -ei quadratic
```

#### Exponential (-exp)

An option designed to favour exponential like emergence, based on a yearly weighting function that linearly increases from zero, for example:

```
python pygrams.py -ts -exp
```

### Timeseries Forecasts

Various options are available to control how emergence is forecasted.

#### Predictor Names (-pns)

The forecast method is selected using argument pns, in this case corresponding to Linear (2=default) and Holt-Winters (6). 

```
Python pygrams.py -pns=2
Python pygrams.py -pns=6
```

The full list of options is included below, with multiple inputs are allowed.

0. All options
1. Naive
2. Linear
3. Quadratic
4. Cubic
5. ARIMA
6. Holt-Winters
7. LSTM-multiLA-stateful
8. LSTM-multiLA-stateless
9. LSTM-1LA-stateful
10. LSTM-1LA-stateless
11. LSTM-multiM-1LA-stateful
12. LSTM-multiM-1LA-stateless

#### Other options

number of terms to analyse (default: 25)

```
Python pygrams.py -nts=25
```

minimum number of patents per quarter referencing a term (default: 15)

```
Python pygrams.py -mpq=15
```

number of steps ahead to analyse for (default: 5) 

```
Python pygrams.py -stp=5
```

analyse using test or not (default: False)

```
Python pygrams.py -tst=False
```

analyse using normalised patents counts or not (default: False)

```
Python pygrams.py -nrm=False
```

### Outputs (-o)

Pygrams outputs a report of top ranked terms (popular or emergent). Additional command line arguments provide alternative options, for example a word cloud or 'graph summary'.

```
python pygrams.py -o wordcloud
python pygrams.py -o graph
```

Time series analysis also supports a multiplot to present up to 30 terms time series (emergent and declining), output in the `outputs/emergence` folder:
```
python pygrams.py -ts -dh 'publication_date' -o multiplot
```

The output options generate:

- Report is a text file containing top n terms (default is 250 terms, see `-np` for more details)
- `wordcloud`: a word cloud containing top n terms (default is 250 terms, see `-nd` for more details)
- `graph`: a summary graph containing top n terms (default is 250 terms, see `-nf` for more details) linked to the most common co-located terms

Note that all outputs are generated in the `outputs` subfolder. Below are some example outputs:

#### Report

The report will output the top n number of terms (default is 250) and their associated TFIDF score. Below is an example for patent data, where only bigrams have been analysed.

|Term			                |	    TFIDF Score     |
|:------------------------- | -------------------:|
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

#### Wordcloud ('wordcloud')

A wordcloud, or tag cloud, is a novel visual representation of text data, where words (tags) importance is shown with font size and colour. Here is a [wordcloud](https://raw.githubusercontent.com/datasciencecampus/pygrams/master/outputs/wordclouds/wordcloud_tech.png) using patent data. The greater the TFIDF score, the larger the font size of the term.

#### Graph summary ('graph')

This output provides an interactive HTML graph. The graph shows connections between terms that are generally found in the same documents.

| Term                       | Connections                                                  |
| -------------------------- | ------------------------------------------------------------ |
| semiconductor substrate    | dielectric layer: 0.16, conductive layer: 0.14, insulating film: 0.14, semiconductor structure: 0.13, channel region: 0.13, gate dielectric layer: 0.12, gate electrode: 0.11, doped region: 0.11, gate stack: 0.10, isolation region: 0.09 |
| pharmaceutical composition | pharmaceutically acceptable salt: 0.82, general formula: 0.22, pharmaceutically acceptable carrier: 0.16, novel compound: 0.16, as define : 0.15, treat disease: 0.15, active ingredient: 0.13, intermediate useful: 0.13, treatment and/or prevention: 0.13, inflammatory disease: 0.13 |
| mobile device              | mobile application: 0.12, mobile device base: 0.12, medium device: 0.08, communication device: 0.08, second mobile device: 0.07, optical imaging lens: 0.06, medium content: 0.06, base station: 0.06, digital content: 0.06, communication network: 0.05 |
| memory cell                | bit line  : 0.61, memory cell array: 0.45, memory device: 0.25, memory array: 0.14, memory block: 0.13, control gate: 0.12, word line : 0.11, flash memory device: 0.11, second memory cell: 0.11, memory cell arrange: 0.10 |


### Folder structure

- pygrams.py is the main python program file in the root folder (Pygrams).
- README.md is this markdown readme file in the root folder
- pipeline.py in the scripts folder provides the main program sequence along with pygrams.py.
- The 'data' folder is where to place the source text data files.
- The 'outputs' folder contains all the program outputs.
- The 'config' folder contains the stop word configuration files.
- The setup file in the root folder, along with the meta folder, contain installation related files.
- The test folder contains unit tests.

## Help

A help function details the range and usage of these command line arguments:

```
python pygrams.py -h
```

The help output is included below. This starts with a summary of arguments:

```
usage: pygrams.py [-h] [-ds DOC_SOURCE] [-it INPUT_TFIDF] [-th TEXT_HEADER]
                  [-dh DATE_HEADER] [-fc FILTER_COLUMNS]
                  [-fb {union,intersection}]
                  [-st SEARCH_TERMS [SEARCH_TERMS ...]]
                  [-stthresh SEARCH_TERMS_THRESHOLD [SEARCH_TERMS_THRESHOLD ...]]
                  [-df DATE_FROM] [-dt DATE_TO] [-mn {1,2,3}] [-mx {1,2,3}]
                  [-mdf MAX_DOCUMENT_FREQUENCY] [-ndl] [-pt PREFILTER_TERMS]
                  [-t] [-o [{graph,wordcloud} [{graph,wordcloud} ...]]]
                  [-on OUTPUTS_NAME] [-wt WORDCLOUD_TITLE] [-nltk NLTK_PATH]
                  [-np NUM_NGRAMS_REPORT] [-nd NUM_NGRAMS_WORDCLOUD]
                  [-nf NUM_NGRAMS_FDG] [-cpc CPC_CLASSIFICATION] [-ts]
                  [-pns PREDICTOR_NAMES [PREDICTOR_NAMES ...]] [-nts NTERMS]
                  [-mpq MINIMUM_PER_QUARTER] [-stp STEPS_AHEAD] [-cf] [-nrm]

extract popular n-grams (words or short phrases) from a corpus of documents
```
It continues with a detailed description of the arguments:
```
  -h, --help            show this help message and exit
  -ds DOC_SOURCE, --doc_source DOC_SOURCE
                        the document source to process (default: USPTO-
                        random-1000.pkl.bz2)
  -it INPUT_TFIDF, --input_tfidf INPUT_TFIDF
                        Load a pickled TFIDF output instead of creating TFIDF
                        by processing a document source (default: None)
  -th TEXT_HEADER, --text_header TEXT_HEADER
                        the column name for the free text (default: abstract)
  -dh DATE_HEADER, --date_header DATE_HEADER
                        the column name for the date (default: None)
  -fc FILTER_COLUMNS, --filter_columns FILTER_COLUMNS
                        list of columns with binary entries by which to filter
                        the rows (default: None)
  -fb {union,intersection}, --filter_by {union,intersection}
                        Returns filter: intersection where all are 'Yes' or
                        '1'or union where any are 'Yes' or '1' in the defined
                        --filter_columns (default: union)
  -st SEARCH_TERMS [SEARCH_TERMS ...], --search_terms SEARCH_TERMS [SEARCH_TERMS ...]
                        Search terms filter: search terms to restrict the
                        tfidf dictionary. Outputs will be related to search
                        terms (default: [])
  -stthresh SEARCH_TERMS_THRESHOLD [SEARCH_TERMS_THRESHOLD ...], --search_terms_threshold SEARCH_TERMS_THRESHOLD [SEARCH_TERMS_THRESHOLD ...]
                        Provides the threshold of how related you want search
                        terms to be Values between 0 and 1: 0.8 is considered
                        high (default: 0.75)
  -df DATE_FROM, --date_from DATE_FROM
                        The first date for the document cohort in YYYY/MM/DD
                        format (default: None)
  -dt DATE_TO, --date_to DATE_TO
                        The last date for the document cohort in YYYY/MM/DD
                        format (default: None)
  -mn {1,2,3}, --min_ngrams {1,2,3}
                        the minimum ngram value (default: 1)
  -mx {1,2,3}, --max_ngrams {1,2,3}
                        the maximum ngram value (default: 3)
  -mdf MAX_DOCUMENT_FREQUENCY, --max_document_frequency MAX_DOCUMENT_FREQUENCY
                        the maximum document frequency to contribute to TF/IDF
                        (default: 0.05)
  -ndl, --normalize_doc_length
                        normalize tf-idf scores by document length (default:
                        False)
  -pt PREFILTER_TERMS, --prefilter_terms PREFILTER_TERMS
                        Initially remove all but the top N terms by TFIDF
                        score before pickling initial TFIDF (removes 'noise'
                        terms before main processing pipeline starts)
                        (default: 100000)
  -o [{graph,wordcloud} [{graph,wordcloud} ...]], --output [{graph,wordcloud} [{graph,wordcloud} ...]]
                        Note that this can be defined multiple times to get
                        more than one output. (default: [])
  -on OUTPUTS_NAME, --outputs_name OUTPUTS_NAME
                        outputs filename (default: out)
  -wt WORDCLOUD_TITLE, --wordcloud_title WORDCLOUD_TITLE
                        wordcloud title (default: Popular Terms)
  -nltk NLTK_PATH, --nltk_path NLTK_PATH
                        custom path for NLTK data (default: None)
  -np NUM_NGRAMS_REPORT, --num_ngrams_report NUM_NGRAMS_REPORT
                        number of ngrams to return for report (default: 250)
  -nd NUM_NGRAMS_WORDCLOUD, --num_ngrams_wordcloud NUM_NGRAMS_WORDCLOUD
                        number of ngrams to return for wordcloud (default:
                        250)
  -nf NUM_NGRAMS_FDG, --num_ngrams_fdg NUM_NGRAMS_FDG
                        number of ngrams to return for fdg graph (default:
                        250)
  -cpc CPC_CLASSIFICATION, --cpc_classification CPC_CLASSIFICATION
                        the desired cpc classification (for patents only)
                        (default: None)
  -ts, --timeseries
                        denote whether emerging technology should be forecast
                        (default: False)
  -pns PREDICTOR_NAMES [PREDICTOR_NAMES ...], --predictor_names PREDICTOR_NAMES [PREDICTOR_NAMES ...]
                        0. All, 1. Naive, 2. Linear, 3. Quadratic, 4. Cubic,
                        5. ARIMA, 6. Holt-Winters, 7. LSTM-multiLA-stateful,
                        8. LSTM-multiLA-stateless, 9. LSTM-1LA-stateful, 10.
                        LSTM-1LA-stateless, 11. LSTM-multiM-1LA-stateful, 12.
                        LSTM-multiM-1LA-stateless; multiple inputs are
                        allowed. (default: [2])
  -nts NTERMS, --nterms NTERMS
                        number of terms to analyse (default: 25)
  -mpq MINIMUM_PER_QUARTER, --minimum-per-quarter MINIMUM_PER_QUARTER
                        minimum number of patents per quarter referencing a
                        term (default: 15)
  -stp STEPS_AHEAD, --steps_ahead STEPS_AHEAD
                        number of steps ahead to analyse for (default: 5)
  -cf, --curve-fitting  analyse using curve or not (default: False)
  -nrm, --normalised    analyse using normalised patents counts or not
                        (default: False)
```

## Acknowledgements

### Patent data

Patent data was obtained from the [United States Patent and Trademark Office (USPTO)](https://www.uspto.gov) through the [Bulk Data Storage System (BDSS)](https://bulkdata.uspto.gov). In particular we used the `Patent Grant Full Text Data/APS (JAN 1976 - PRESENT)` dataset, using the data from 2004 onwards in XML 4.* format.

### scikit-learn usage

Sections of this code are based on [scikit-learn](https://github.com/scikit-learn/scikit-learn) sources.

### Knockout JavaScript library

The [Knockout](http://knockoutjs.com/) JavaScript library is used with our force-directed graph output.

### WebGenresForceDirectedGraph

The [WebGenresForceDirectedGraph](https://github.com/Aeternia-ua/WebGenresForceDirectedGraph) 
project by Iryna Herasymuk is used to generate the force directed graph output.

### 3rd Party Library Usage

Various 3rd party libraries are used in this project; these are listed
on the [dependencies](https://github.com/datasciencecampus/pygrams/network/dependencies) page, whose contributions we gratefully acknowledge. 

