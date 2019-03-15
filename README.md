[![build status](http://img.shields.io/travis/datasciencecampus/pyGrams/master.svg?style=flat)](https://travis-ci.org/datasciencecampus/pyGrams)
[![Build status](https://ci.appveyor.com/api/projects/status/oq49c4xuhd8j2mfp/branch/master?svg=true)](https://ci.appveyor.com/project/IanGrimstead/patent-app-detect/branch/master)
[![codecov](https://codecov.io/gh/datasciencecampus/pyGrams/branch/master/graph/badge.svg)](https://codecov.io/gh/datasciencecampus/pyGrams)
[![LICENSE.](https://img.shields.io/badge/license-OGL--3-blue.svg?style=flat)](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)

# pyGrams 

<p align="center"><img align="center" src="meta/images/pygrams-logo.png" width="400px"></p>

## Description of tool

This python-based app (`pygrams.py`) is designed to extract popular n-grams (words or short phrases) from free text within a large (>1000) corpus of documents. Example corpora of patent document abstracts are included for testing.

The app operates in the following steps:

- A file containing a corpora of documents (placed in the /data folder) is selected (defaulting to a 1000 abstract patent file), where each row or list element in a file corresponds to a document. The column for the text to be analysed is specified, and optionally the rows can be filtered by date and by binary entries in specified columns.
- The core function of the app is to perform TFIDF on the document corpus, optionally specifying minimum and maximum ngrams, and maximum document frequency. The resulting TDIDF matrix is stored on file.
- The TFIDF matrix may subsequently be post processed using a mask comprising document weight vectors and term weight vectors. Document weightings include document length normalisation and time weighting (more recent documents weighted more highly). Term weightings include stop words, and word embeddings.
- The default 'report' output is a ranked and scored list of 'popular' ngrams. Optional outputs are a graph, word cloud, tfidf matrix, and terms counts.

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
   pip install -e
   ```

   This will install all the libraries and run some tests. If the tests pass, the app is ready to run. If any of the tests fail, please email [ons.patent.explorer@gmail.com](mailto:ons.patent.explorer@gmail.com) with a screenshot of the failure so that we may get back to you, or alternatively open a [GitHub issue here](https://github.com/datasciencecampus/pyGrams/issues).

### System requirements

We have stress-tested `pygrams.py` using Windows 10 (64-bit) with 8GB memory (VM hosted on 2.1GHz Xeon E5-2620). We observed a linear increase in both execution time and memory usage in relation to number of documents analysed, resulting in:

- Processing time: 41.2 documents/sec
- Memory usage: 236.9 documents/MB

For the sample files, this was recorded as:

- 1,000 documents: 0:00:37
- 10,000 documents: 0:04:45 (285s); 283MB
- 100,000 documents: 0:40:10 (2,410s); 810MB
- 500,000 documents: 3:22:08 (12,128s); 2,550MB

## User guide

pyGrams is command line driven, and called in the following manner:

```
python pygrams.py
```

### Document Parameters

#### Selecting the document source (-ds, -pt)

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

To use your own document dataset, either place in the `./data` folder of pyGrams or change the path using `-pt`. File types currently supported are:

- pkl.bz2: compressed pickle file containing a dataset
- xlsx: new Microsoft excel format
- xls: old Microsoft excel format
- csv: comma separated value file (with headers)

Datasets should contain the following columns:

|Column			            |	    Required?  |
| :------------------------ | -------------------:|
|Unique ID                  |       Yes           |
|Free text field            |       Yes           |
|Date                       |       Optional      |
|Boolean fields (Yes/No)    |       Optional      |

The unique ID field should contain unique identifiers for each row of the dataset. The free text field can be any free text, for example an abstract. The date field should be in the format `YYYY-MM-DD HH:MM:SS`. The boolean fields can be any Yes/No data (there may be multiple)

#### Selecting the column header names (-nh, -th, -dh)

When loading a document dataset, you will need to provide the column header names for each, using:

- `-nh`: unique ID column (default is 'id')
- `-th`: free text field column (default is 'text')
- `-dh`: date column (default is 'date')

For example, for a corpus of book blurbs you could use:

```
python pygrams.py -nh='book_name' -th='blurb' -dh='published_date'
```

#### Word filters (-fh, -fb)

If you want to filter results, such as for female, British, authors in the above example, you can specify the boolean (yes/no) column names you wish to filter by, and the type of filter you want to apply, using:

- `-fh`: the list of boolean fields (default is None)
- `-fb`: the type of filter (choices are `'union'` (default), where all fields need to be 'yes', or `'intersection'`, where any field can be 'yes') 

```
python pygrams.py -fh=['female','british'] -fb='union'
```

#### Time filters (-mf, -yf, -mt, -yt)

This argument can be used to filter documents to a certain timeframe. For example, the below, will restrict the document cohort to only those from 2000 up to now (the default 'month from' `-mf` is January).

```
python pygrams.py -yf=2000
```

This will restrict the document cohort to only those between March 2000 and July 2016.

```
python pygrams.py -mf=03 -yf=2000 -mt=07 -yt=2016
```

### TF-IDF Parameters 

#### N-gram selection (-mn, -mx)

An n-gram is a contiguous sequence of n items ([source](https://en.wikipedia.org/wiki/N-gram)). N-grams can be unigrams (single words, e.g., vehicle), bigrams (sets of words, e.g., aerial vehicle), trigrams (trio of words, e.g., unmanned aerial vehicle) or any n number of continuous terms. 

The following arguments will set the n-gram limit to be bigrams or trigrams (the default).

```
python pygrams.py -mn=2 -mx=3
```

To analyse only unigrams:

```
python pygrams.py -mn=1 -mx=1
```

#### Maximum document frequency (-mdf)

Terms identified are filtered by the maximum number of documents that use this term; the default is 0.3, representing an upper limit of 30% of documents containing this term. If a term occurs in more that 30% of documents it is rejected.

For example, to set the maximum document frequency to 5%, use:

```
python pygrams.py -mdf 0.05
```

By using a small (5% or less) maximum document frequency for unigrams, this may help remove generic words, or stop words.

#### TF-IDF score mechanics (-p)

By default the TF-IDF score will be calculated per n-gram as the sum of the TF-IDF values over all documents for the selected n-gram. However you can select:

- `median`: the median value
- `max`: the maximum value
- `sum`: the sum of all values
- `avg`: the average, over non zero values

To choose an average scoring for example, use:

```
python pygrams.py -p='avg'
```

#### Normalise by document length (-ndl)

This option normalises the TF-IDF scores by document length.

```
python pygrams.py -ndl
```

#### Time-weighting (-t)

This option applies a linear weight that starts from 0.01 and ends at 1 between the time limits.

```
python pygrams.py -t
```

### Outputs Parameters (-o)

The default option outputs a report of top ranked terms. Additional command line arguments provide alternative options, for example a word cloud or force directed graph (fdg) output. The option 'all', produces all:

```
python pygrams.py -o='report'
python pygrams.py -o='wordcloud'
python pygrams.py -o='fdg'
python pygrams.py -o='table'
python pygrams.py -o='tfidf'
python pygrams.py -o='termcounts'
python pygrams.py -o='all'
```

The output options generate:

- `report` (default): a text file containing top n terms (default is 250 terms, see `-np` for more details)
- `wordcloud`: a word cloud containing top n terms (default is 250 terms, see `-nd` for more details)
- `fdg`: a force-directed graph containing top n terms (default is 250 terms, see `-nf` for more details)
- `table`: an XLS spreadsheet to compare term rankings
- `tfidf`: a pickle of the TFIDF matrix
- `termcounts`: a pickle of term counts per week
- `all`: all of the above

Note that all outputs are generated in the `outputs` subfolder. Below are some example outputs:

#### Report ('report')

The report will output the top n number of terms (default is 250) and their associated [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) score. Below is an example for patent data, where only bigrams have been analysed.

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

#### Wordcloud ('wordcloud')

A wordcloud, or tag cloud, is a novel visual representation of text data, where words (tags) importance is shown with font size and colour. Here is a [wordcloud](https://raw.githubusercontent.com/datasciencecampus/pygrams/master/outputs/wordclouds/wordcloud_tech.png) using patent data. The greater the TF-IDF score, the larger the font size of the term.

#### Force directed graph ('fdg')

This output provides an interactive HTML graph. The graph shows connections between terms that are generally found in the same documents.

#### TF-IDF matrix ('tfidf')

The TF-IDF matrix can be saved as a pickle file, containing a list of four items:

- The TF-IDF sparse matrix
- List of unique terms
- List of document publication dates
- List of unique document IDs

#### Term counts matrix

Of use for further processing, the number of patents containing a term in a given week
(stored as a matrix) can be output as a pickle file, containing a list of four items:
- Term counts per week (sparse matrix)
- List of terms (column heading)
- List of number of patents in that week
- List of patent publication dates (row heading)

### Patent specific support

#### Choosing CPC classification

This subsets the chosen patents dataset to a particular Cooperative Patent Classification (CPC) class, for example Y02. 
The Y02 classification is for "technologies or applications for mitigation or adaptation against climate change". In 
this case a larger patent dataset is generally required to allow for the reduction in patent numbers after subsetting. 
An example script is:

```
python pygrams.py -cpc=Y02 -ps=USPTO-random-10000.pkl.bz2
```

In the console the number of subset patents will be stated. For example, for `python pygrams.py -cpc=Y02 -ps=USPTO-random-10000.pkl.bz2` the number of Y02 patents is 197. Thus, the tf-idf will be run for 197 patents.

#### Citation weighting

This will weight the term TFIDF scores by the number of citations each patent has. The weight is a normalised value between 0 and 1 with the higher the number indicating a higher number of citations.

```
python pygrams.py -c
```

### Config files

There are three configuration files available inside the config directory:

- stopwords_glob.txt
- stopwords_n.txt
- stopwords_uni.txt

The first file (stopwords_glob.txt) contains stopwords that are applied to all n-grams.
The second file contains stopwords that are applied to all n-grams for n > 1 (bigrams and trigrams) and the last file (stopwords_uni.txt) contains stopwords that apply only to unigrams. The users can append stopwords into this files, to stop undesirable output terms.

### Folder structure

- pygrams.py is the main python program file in the root folder (Pygrams).
- README.md is this markdown readme file in the root folder
- pipeline.py in the scripts folder provides the main program sequence along with pygrams.py.
- The data folder is where to place the source text data files.
- The outputs folder contains all the program outputs.
- The config folder contains the stop word configuration files.
- The setup file in the root folder, along with the meta folder, contain installation related files.
- The test folder contains unit tests.

## Help

A help function details the range and usage of these command line arguments:

```
python pygrams.py -h
```

An edited version of the help output is included below. This starts with a summary of arguments:

```
python pygrams.py -h
usage: pygrams.py [-h] [-cpc CPC_CLASSIFICATION] [-c] [-f {set,chi2,mutual}]
                  [-ndl] [-t] [-pt PATH] [-ih ID_HEADER] [-th TEXT_HEADER]
                  [-dh DATE_HEADER] [-fc FILTER_COLUMNS]
                  [-fb {union,intersection}] [-p {median,max,sum,avg}]
                  [-o {fdg,wordcloud,report,table,tfidf,termcounts,all}] [-j]
                  [-yf YEAR_FROM] [-mf MONTH_FROM] [-yt YEAR_TO]
                  [-mt MONTH_TO] [-np NUM_NGRAMS_REPORT]
                  [-nd NUM_NGRAMS_WORDCLOUD] [-nf NUM_NGRAMS_FDG]
                  [-ds DOC_SOURCE] [-fs FOCUS_SOURCE] [-mn {1,2,3}]
                  [-mx {1,2,3}] [-mdf MAX_DOCUMENT_FREQUENCY]
                  [-rn REPORT_NAME] [-wn WORDCLOUD_NAME] [-wt WORDCLOUD_TITLE]
                  [-tn TABLE_NAME] [-nltk NLTK_PATH]

create report, wordcloud, and fdg graph for document abstracts

```
It continues with a detailed description of the arguments:
```
optional arguments:
  -h, --help            show this help message and exit
  -cpc CPC_CLASSIFICATION, --cpc_classification CPC_CLASSIFICATION
                        the desired cpc classification (for patents only)
  -c, --cite            weight terms by citations (for patents only)
  -f {set,chi2,mutual}, --focus {set,chi2,mutual}
                        clean output from terms that appear in general; 'set':
                        set difference, 'chi2': chi2 for feature importance,
                        'mutual': mutual information for feature importance
  -ndl, --normalize_doc_length
                        normalize tf-idf scores by document length
  -t, --time            weight terms by time
  -pt PATH, --path PATH
                        the data path
  -ih ID_HEADER, --id_header ID_HEADER
                        the column name for the unique ID
  -th TEXT_HEADER, --text_header TEXT_HEADER
                        the column name for the free text
  -dh DATE_HEADER, --date_header DATE_HEADER
                        the column name for the date
  -fc FILTER_COLUMNS, --filter_columns FILTER_COLUMNS
                        list of columns to filter by
  -fb {union,intersection}, --filter_by {union,intersection}
                        options are <all> <any> defaults to any. Returns
                        filter where all are 'Yes' or any are 'Yes
  -p {median,max,sum,avg}, --pick {median,max,sum,avg}
                        options are <median> <max> <sum> <avg> defaults to
                        sum. Average is over non zero values
  -o {fdg,wordcloud,report,table,tfidf,termcounts,all}, --output {fdg,wordcloud,report,table,tfidf,termcounts,all}
                        options are: <fdg> <wordcloud> <report> <table>
                        <tfidf> <termcounts> <all>
  -j, --json            Output configuration as JSON file alongside output
                        report
  -yf YEAR_FROM, --year_from YEAR_FROM
                        The first year for the document cohort in YYYY format
  -mf MONTH_FROM, --month_from MONTH_FROM
                        The first month for the document cohort in MM format
  -yt YEAR_TO, --year_to YEAR_TO
                        The last year for the document cohort in YYYY format
  -mt MONTH_TO, --month_to MONTH_TO
                        The last month for the document cohort in MM format
  -np NUM_NGRAMS_REPORT, --num_ngrams_report NUM_NGRAMS_REPORT
                        number of ngrams to return for report
  -nd NUM_NGRAMS_WORDCLOUD, --num_ngrams_wordcloud NUM_NGRAMS_WORDCLOUD
                        number of ngrams to return for wordcloud
  -nf NUM_NGRAMS_FDG, --num_ngrams_fdg NUM_NGRAMS_FDG
                        number of ngrams to return for fdg graph
  -ds DOC_SOURCE, --doc_source DOC_SOURCE
                        the doc source to process
  -fs FOCUS_SOURCE, --focus_source FOCUS_SOURCE
                        the doc source for the focus function
  -mn {1,2,3}, --min_n {1,2,3}
                        the minimum ngram value
  -mx {1,2,3}, --max_n {1,2,3}
                        the maximum ngram value
  -mdf MAX_DOCUMENT_FREQUENCY, --max_document_frequency MAX_DOCUMENT_FREQUENCY
                        the maximum document frequency to contribute to TF/IDF
  -rn REPORT_NAME, --report_name REPORT_NAME
                        report filename
  -wn WORDCLOUD_NAME, --wordcloud_name WORDCLOUD_NAME
                        wordcloud filename
  -wt WORDCLOUD_TITLE, --wordcloud_title WORDCLOUD_TITLE
                        wordcloud title
  -tn TABLE_NAME, --table_name TABLE_NAME
                        table filename
  -nltk NLTK_PATH, --nltk_path NLTK_PATH
                        custom path for NLTK data
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

