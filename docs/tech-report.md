![img](img/pygrams-logo-3.png)

## Team:

- Thanasis Anthopoulos: thanasis.anthopoulos@ons.gov.uk
- Ian Grimstead: ian.grimstead@ons.gov.uk
- Michael Hodge: michael.hodge@ons.gov.uk
- Bernard Peat: bernard.peat@ons.gov.uk
- Emily Tew: emily.tew@ons.gov.uk
- Fatima Chiroma

# Objectives and scope 1-2
## Customers
### Ipo
### Dirac
### People survey
## Previous and related work

# Data engineering 1-2 I

To enable the most generic support possible at minimum overhead, we decided to store input data using
[Pandas](https://pandas.pydata.org/) dataframes; each text sample then becomes a row in a dataframe
(which is comparable to a database table, only stored directly in memory rather than on disc).

## Patent Data

Initially, we did not have access to [PatStat](https://www.epo.org/searching-for-patents/business/patstat.html#tab-1)
(the world-wide patent archive), but were given access to samples from the UK's patent data in XML format. To enable
us to use large numbers of patent abstract as soon as possible, we imported the USPTO's
[bulk patent](https://bulkdata.uspto.gov/) dataset, using data from 2014 onwards (as this was
stored in a similar XML format). The XML data was scraped from the web using
[beautifulsoup](https://www.crummy.com/software/BeautifulSoup/) and exported in data frame
format for ingestion into pygrams.

Later, when patstat became available, we created an import tool which parsed the CSV format
data supplied by patstat and directly exported in dataframe format, to avoid the need for an intermediate
database.

# Data sources 1-2

Suggest this is folded into s/w engineering section.

# Key terms extraction 7

When you type text into a computer it can't understand the words in the way that humans can. Everytime a character is typed that character is converted into a binary number that the computer can read but doesn't assign any meaning to. That said, the word *'key'* in *'key terms'* implies the computer needs to have some concept of 'meaning' to identify terms as *'key'*. The branch of Data Science responsible for processing and analysing language in this way is known as **Natural Language Processing (NLP)** and it provides many tools that Data Scientists can use to extract meaning from text data.



## Previous and related work
## TFIDF

**PyGrams** uses a tool called Term Frequency - Inverse Document Frequency or **TF-IDF** for short.

TF-IDF is a widely used technique to retrieve key words (or in our case, terms) from a corpus. The output of TF-IDF is a TF-IDF weight which can be used to rank the importance of terms within the corpus.  It can be broken down into two parts:

1. **Term Frequency (TF)** = **$\frac{\text{The number of times a term appears in a document}}{\text{Total number of terms in the document}}$**

<br/>

2. **Inverse Document Frequency (IDF)** = **$\ln(\frac{\text{Total number of documents}}{\text{Number of documents which contains the term}})$**

<br/>

For example, lets say Document 1 contains 200 terms and the term *'nuclear'* appears 5 times.

**Term Frequency** = **$\frac{5}{200}$** = 0.025

 Also, assume we have 20 million documents and the term *'nuclear'* appears in ten thousand of these.

**Inverse Document Frequency** = $\ln(\frac{20,000,000}{10,000})$ = 7.6

 Therefore, **TF-IDF weight** for *'nuclear'* in Document 1 = $0.025 \times7.6 = 0.19$.

 Eventually this produces a matrix of TF-IDF weights which are summed to create the final TFIDF weight:

 | Document_no  | 'nuclear'  | 'electric'  | 'people'  |
|:---:|:---:|:---:|:---:|
|  1 | 0.19  | 0.11  |  0.10 |
|  2 |  0.22 |  0.02 |  0.12 |
|  3 |  0.17 |  0.04 |  0.13 |
|  **Final_Weight**  |   **0.58**    | **0.17**  | **0.35**  |
## Filtering 2
### CPC
### Stop words
#### Manual list
#### Fatima work
### Word embedding 1 E

The terms filter in PyGrams is used to filter out terms which are not relevant to terms inputted by the user. To do this, it uses a GloVe pre-trained word embedding.

#### What is a GloVe pre-trained word embedding?

**GloVe is an unsupervised learning algorithm for obtaining vector representations for words.** For a model to be 'pre-trained' the algorithm needs to be trained on a corpus of text where it learns to produce word vectors that are meaningful given the word's co-occurance with other words. Once the word vectors have been learnt the Euclidean distance between them can be used to measure semantic similarity of the words.

Below is a visual representation of a vector space for the term MEMS  (Micro-Electro-Mechanical Systems):

![img](img/embedding.png)

The model used for PyGrams has been trained on a vocabulary of 400,000 words from Wikipedia 2014 and an archive of newswire text data called Gigaword 5. For our purposes, the 50 dimensional vector is used to reduce the time it takes for the filter to run (particularly with a large dataset).

All GloVe word vectors can be downloaded [here](https://nlp.stanford.edu/projects/glove/).

#### How does it learn word vectors?

Unlike other embedding models, GloVe is a count-based model meaning it is a based on a counts matrix of co-occuring words where the rows are words and the columns are context words. The rows are then factorized to a lower dimensionality (in our case 50) to yield a vector representation that is able to explain the variance in the high dimensionality vector.

The steps go as follows:

##### Step 1: Counts matrix

- Collect counts of co-occuring words and record them in matrix $X$.
- Each cell, $X_{ij}$, refers to how often word $i$ occurs with word $j$ using a pre-defined window size before and after the word.

##### Step 2: Soft Constraints

- For each word pair define soft contraints as follows:
</b>
    $w_{i}^Tw_j + b_i + b_j = \text{log}(X_{ij})$

where:
- $w_i$ is the vector for the main word.
- $w_j$ is the vector for the context word.
- $b_i$ and $b_j$ are biases for the main and context words respectively.

##### Step 3: Cost Function

$ J = \sum_{i=1}^V\sum_{j=1}^V f(X_{ij})(w_{i}^Tw_j + b_i + b_j = \text{log}(X_{ij})^2$

where:
- $V$ is the size of the vocabulary.
- $f$ is a weighting function to prevent overweighting the common word pairs.

#### How does it work in PyGrams?

Given a distance threshold between user inputted words and words in the corpus, words that are within the threshold distance are included in the output and those which are not are excluded.

PyGrams does this by using a TFIDF mask. As previously mentioned the TFIDF is used to filter out important words in the corpus and is stored in a TFIDF matrix. The vectors for the user inputted words are compared to the vectors of the words in the TFIDF matrix and those within a given threshold are kept in the output.


The above functionality is acheived using the following piece of code:

    def __get_embeddings_vec(self, threshold):
        embeddings_vect = []
        for term in tqdm(self.__tfidf_ngrams, desc='Evaluating terms distance with: ' + ' '.join(self.__user_ngrams), unit='term',
                         total=len(self.__tfidf_ngrams)):
            compare = []
            for ind_term in term.split():
                for user_term in self.__user_ngrams:
                    try:
                        similarity_score = self.__model.similarity(ind_term, user_term)
                        compare.append(similarity_score)
                    except:
                        compare.append(0.0)
                        continue

            max_similarity_score = max(similarity_score for similarity_score in compare)
            embeddings_vect.append(max_similarity_score)
        #embeddings_vect_norm = ut.normalize_array(embeddings_vect, return_list=True)
        if threshold is not None:
            return [float(x>threshold) for x in embeddings_vect]
        return embeddings_vect


#### What is the output?

Using a random selection of 1,000 patents from USPTO and running the following code:

        python pygrams.py

the following terms came out as top:

    1. semiconductor substrate        1.678897
    2. electronic device              1.445151
    3. semiconductor device           1.347494
    4. liquid crystal display         1.202028
    5. top surface                    1.158683
    6. pharmaceutical composition     1.074344
    7. corn plant                     1.073920
    8. longitudinal axis              1.067497
    9. light source                   1.054303
    10. power supply                   0.953445


Using the same dataset but adding a terms filter for medical words and a threshold of 0.8:

        python pygrams.py -st pharmacy medical hospital

the following terms came out as top:

    1. medical data                   0.414319
    2. subsequent treatment           0.257790
    3. treatment objective            0.213937
    4. ink-repellent treatment        0.211789
    5. clinical analysis system       0.192105
    6. heat treatment apparatus       0.187454
    7. radiation-delivery treatment plan 0.186285
    8. biocompatibilize implantable medical 0.156199
    9. implantable medical device     0.156199
    10. cardiovascular health          0.153314



To find out how to run term filtering in PyGrams please see the 'Term Filter' section in the PyGrams README found on [Github](https://github.com/datasciencecampus/pyGrams#term-filters)
## Tfidf 2 E
### Weightings E
#### Citations M
## Outputs 2 IT
### Fog
### Word cloud
# Time series 4
## Previous and related work
## Escores 2 IT
### Porter
### Curves
## Prediction 2 IB
### LSTM
### Arima
### Holt winters
### Quad cubic etc
# Conclusion 1
# References
