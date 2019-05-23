# Style guide

Use large USPTO pickle which has been prefiltered for all results - reproducable. DataFrame will be pickled and stored
in GitHub. Example; ```python pygrams.py -it USPTO-mdf-0.05

Write up all commands against the results to show how it worked!


![img](img/pygrams-logo-3.png)

## Team:

- Thanasis Anthopoulos: thanasis.anthopoulos@ons.gov.uk
- Fatima Chiroma
- Ian Grimstead: ian.grimstead@ons.gov.uk
- Michael Hodge: michael.hodge@ons.gov.uk
- Sonia Mazi sonia.mazi@ons.gov.uk
- Bernard Peat: bernard.peat@ons.gov.uk
- Emily Tew: emily.tew@ons.gov.uk

# Objectives and scope 1-2 T
The present project aimed in generating insights out of large document collections. By large document collections we 
mean a large number of documents ( >=10000 ) that share the same theme, like patents, job adverts, medical journal 
publications and others. The insights we are aiming to retrieve from these document collections are:
- popular terminology
- emerging terminology

Popular terminology refers to the most frequent keywords and small phrases ( up  to three words) and emerging 
terminology is keywords that show emerging ( or declining ) frequency patterns when projected on a time-series scale.

## Stakeholders
Initially this project idea came from BEIS and the IPO, where the former was popular key-terminology to be 
retrieved from patent applications and the latter came with the idea of retrieving emerging terminology. Both approaches
would aim in providing richer information for various technology sectors for policy. The list below demonstrates the 
various stakeholders that have expressed an interest in using our pipeline for similar datasets since we started working
on this project.
- IPO: Emerging terminology from patent data (PATSTAT)
- BEIS: popular terminology from UK patents
- ONS:
    - popular terminology on people survey free-text comments
    - Emerging terminology on statistical journal publications data
    - Emerging terminology on coroners reports
- DIRAC: Emerging terminology in job adverts. Identification of emerging job skills
- Innovate UK: Emerging and popular terminology on project grant applications
- MOJ: popular terminology on people survey free-text comments
- GDS: Emerging terminology in job adverts. Identification of emerging job skills in DDaT profession
- DIT: Popular terminology on EU Exit consultations

# Data engineering 1-2 I

To enable the most generic support possible at minimum overhead, we decided to store input data using
[Pandas](https://pandas.pydata.org/) dataframes; each text sample then becomes a row in a dataframe
(which is comparable to a database table, only stored directly in memory rather than on disc).

## Patent Data

Initially, we did not have access to [PatStat](https://www.epo.org/searching-for-patents/business/patstat.html#tab-1)
(the world-wide patent archive), but were given access to samples from the UK's patent data in XML format. To enable
us to use large numbers of patent abstract as soon as possible, we imported the USPTO's
[bulk patent](https://bulkdata.uspto.gov/) dataset, using data from 2004 onwards (as this was
stored in a similar XML format). The XML data was scraped from the web using
[beautifulsoup](https://www.crummy.com/software/BeautifulSoup/) and exported in data frame
format for ingestion into pygrams.

Later, when patstat became available, we created an import tool which parsed the CSV format
data supplied by patstat and directly exported in dataframe format, to avoid the need for an intermediate
database.

# Data sources 1-2

Suggest this is folded into s/w engineering section.

# Objective 1: Popular Terminology 7 ETF

When you type text into a computer it can't understand the words in the way that humans can. Everytime a character is typed that character is converted into a binary number that the computer can read but doesn't assign any meaning to. That said, the word *'key'* in *'key terms'* implies the computer needs to have some concept of 'meaning' to identify terms as *'key'*. The branch of Data Science responsible for processing and analysing language in this way is known as **Natural Language Processing (NLP)** and it provides many tools that Data Scientists can use to extract meaning from text data.



## Previous and related work
## Tfidf 2 E

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

## Issues when using mixed length phrases
There are some issues when using mixed length phrases. That is for a given tri-gram ie. combustion engine, its associated 
bi-grams 'internal combustion' and 'combustion engine' as well as its unigrams 'internal', 'combustion' and 'engine'
will receive counts too. So as a post-processing step, we deduct the higher-gram counts from the lower ones in order to
have a less biased output of phrases as a result.

### Dictionary Post process and Reduction
The TFIDF sparse matrix grows exponentially when bi-grams and tri-grams are included. The dictionary of phrases on the
columns of the matrix can quickly grow into tens of millions. This has major storage and performance implications and
was one of the major challenges for this project. In order to allow for faster processing and greater versatility in 
terms of computer specifications needed to run the pygrams pipeline we came up with some optimization ideas.
We decided to discard non-significant features from the matrix and cache it along with the document dates in numerical 
format. 
The matrix optimization is performed by choosing the top n phrases (uni-bi-tri-grams) where n is user 
configurable and defaults to 100,000. The top n phrases are ranked by their sum of tfidf over all documents. In order to
reduce the final object size, we decided to store the term-count matrix instead of the tf-idf as this would mean that we
could use uint8 ie. 1 byte instead of the tf-idf data, which defaults to float64 and is 8 bytes per non-zero data 
element. When the cached object is read back, it only takes linear time to calculate and apply the weights This reduces 
the size of the cached serialized object by a large factor, which means that it can be de-serialized faster when read back.
This way we managed to store 3.2M  US patent data documents in just 56.5 Mb with bz2 compression. This file is stored on
our github page in https://github.com/datasciencecampus/pyGrams/tree/develop/outputs/tfidf/USPTO-mdf-0.05 and has been 
used to produce all the results in this report. We also append the command line arguments used to generate our outputs
so that readers can reproduce them if they wish. The time it takes to cache the object is six and a half hours on a 
macbook pro with 16GB of RAM and i7 cores. Subsequent queries run in the order of one minute for popular terminology and
a few minutes ( 7-8 mins) for timeseries outputs without forecasting.


### Document filtering 0.5-1 B
Once the cached object is read we filter rows and columns based on the user query in order to produce the right results

Document filtering comprises:

- Time filters, restricting the corpus to documents with publication dates within a specified range.
- Column filters, restricting the corpus to documents where the values of selected columns meet specified (binary) criteria. 
For patent data specifically, documents can be restricted to those with a specified Cooperative Patent Classification (CPC) value.

### Term filtering 2 B


#### Stopwords

Stopwords are handled using three user configurable files. One contains global stopwords, including a list of standard 
English stopwords; one contains unigram stop words; and the third bi-gram or tri-gram stopwords.

#### Fatima work TF

#### Word embedding 1 E

The terms filter in PyGrams is used to filter out terms which are not relevant to terms inputted by the user. To do this,
it uses a GloVe pre-trained word embedding. However, our pipeline can be used with other models like word2vec or fasttext.
Glove has been chosen for practical purposes as it is low in storage and fast on execution.

##### What is a GloVe pre-trained word embedding?

**GloVe is an unsupervised learning algorithm for obtaining vector representations for words.** For a model to be 'pre-trained' the algorithm needs to be trained on a corpus of text where it learns to produce word vectors that are meaningful given the word's co-occurance with other words. Once the word vectors have been learnt the Euclidean distance between them can be used to measure semantic similarity of the words.

Below is a visual representation of a vector space for the term MEMS  (Micro-Electro-Mechanical Systems):

![img](img/embedding.png)

The model used for PyGrams has been trained on a vocabulary of 400,000 words from Wikipedia 2014 and an archive of newswire text data called Gigaword 5. For our purposes, the 50 dimensional vector is used to reduce the time it takes for the filter to run (particularly with a large dataset).

All GloVe word vectors can be downloaded [here](https://nlp.stanford.edu/projects/glove/).

##### How does it learn word vectors?

Unlike other embedding models, GloVe is a count-based model meaning it is a based on a counts matrix of co-occuring words where the rows are words and the columns are context words. The rows are then factorized to a lower dimensionality (in our case 50) to yield a vector representation that is able to explain the variance in the high dimensionality vector.

The steps go as follows:

###### Step 1: Counts matrix

- Collect counts of co-occuring words and record them in matrix $X$.
- Each cell, $X_{ij}$, refers to how often word $i$ occurs with word $j$ using a pre-defined window size before and after the word.

###### Step 2: Soft Constraints

- For each word pair define soft contraints as follows:
</b>
    $w_{i}^Tw_j + b_i + b_j = \text{log}(X_{ij})$

where:
- $w_i$ is the vector for the main word.
- $w_j$ is the vector for the context word.
- $b_i$ and $b_j$ are biases for the main and context words respectively.

###### Step 3: Cost Function

$ J = \sum_{i=1}^V\sum_{j=1}^V f(X_{ij})(w_{i}^Tw_j + b_i + b_j = \text{log}(X_{ij})^2$

where:
- $V$ is the size of the vocabulary.
- $f$ is a weighting function to prevent overweighting the common word pairs.

##### How does it work in PyGrams?

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


##### What is the output?

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



To find out how to run term filtering in PyGrams please see the 'Term Filter' section in the PyGrams README found on 
[Github](https://github.com/datasciencecampus/pyGrams#term-filters)

Example output:

command: -it=USPTO-mdf-0.05 -st pharmacy medicine chemist -on=out-embed | exec time: 1:18 mins

1. pharmaceutical composition     		2446.811441
2. medical device                 		847.004068
3. implantable medical device     		376.653856
4. treat cancer                   		303.083392
5. treat disease                  		291.440180
6. work piece                     		279.293397
7. heat treatment                 		278.582799
8. food product                   		245.006287
9. work surface                   		212.500359
10. work fluid                     		202.991614
11. patient body                   		196.159483
12. pharmaceutical formulation     		177.029507
13. treatment and/or prevention   		168.678058
14. cancer cell                    		159.921436
15. provide pharmaceutical composition 	152.719985
16. treatment fluid                		132.922168
17. medical image                  		127.059351
18. medical instrument             		123.362187
19. treatment and/or prophylaxis   		114.959887
20. prostate cancer                		108.456828


cmd: -it=USPTO-mdf-0.05 | exec time: 52 secs

1. semiconductor device           		3181.175539
2. electronic device              		2974.360838
3. light source                   		2861.643506
4. semiconductor substrate        		2602.684013
5. mobile device                  		2558.832724
6. pharmaceutical composition 		    2446.811441
7. electrically connect           		2246.935926
8. base station                   		2008.353328
9. memory cell                    		1955.181403
10. display device                 		1939.361315
11. image data                     		1807.067937
12. main body                      		1799.963480
13. dielectric layer               		1762.330106
14. semiconductor layer            		1749.876921
15. control unit                   		1730.955634
16. circuit board                  		1696.772008
17. control signal                 		1669.367299
18. top surface                    		1659.069068
19. gate electrode                 		1637.708834
20. input signal                   		1567.315205


# Objective 2: Emerging Terminology 4
In order to assess emergence, our dataset needs to be converted into a time-series. Our approach was to reduce the 
tfidf matrix into a timeseries matrix where each term is receiving a document count over a period. For example, if the 
period we set is a month and term 'fuel cell' had a non-zero tfidf for seventeen documents it would get a count of 
seventeen for this month. Once we obtain the timeseries matrix, we benchmarked three different methods to retrieve 
emerging terminology. These were Porter(2018), curve fitting and a state-space model with kalman filter.


## Escores 2 IT
## Previous and related work / Porter
Our first attempts to generate emerging terminology insights were based on the Porter(2018) publication. This method 
relied on ten timeseries periods, the three first being the base period and the following seven the active one. The 
emergence score is calculated using a series of differential equations within the active period counts, normalised by
the global trend.

TODO: replace this with math:

        active_period_trend = (sum_term_counts_567 / sum_sqrt_total_counts_567) - (sum_term_counts_123 / 
        sum_sqrt_total_counts_123)

        recent_trend = 10 * (
                (term_counts[5] + term_counts[6]) / (sqrt(total_counts[5]) + sqrt(total_counts[6]))
                - (term_counts[3] + term_counts[4]) / (sqrt(total_counts[3]) + sqrt(total_counts[4])))

        mid_year_to_last_year_slope = 10 * (
                (term_counts[6] / sqrt(total_counts[6])) - (term_counts[3] / sqrt(total_counts[3]))) / 3

        e_score=  2 * active_period_trend + mid_year_to_last_year_slope + recent_trend

![img](img/porter_2018.png)

This method works well for terms rapidly emerging in the last three periods as it is expected looking at the equations.
It also takes into consideration the global trend, which sometimes may not be desirable 

### Curves
We decided to investigate alternative methods that would be more generic in the sense that emergence could be 
scored uniformly in the given timeseries and normalization by the global trend would be optional. Our immediate next 
thought was to fit second degree polynomials and sigmoid curves to retrieve emerging patterns in our corpus. 

![img](img/curves.png)

Initially we were fitting both sigmoid and polynomial curves and pick the best fit one. However in most cases polynomials
would fit best, so we decided in favour of keeping just them. The e-score is simply the coefficient of the highest rank 
term of the fitted curve, which characterises the slope. For a quadratic, y=ax^2 +bx + c, a determines how steeply the
series emerge (or decline if a is negative).

The emergeThe results were comparable to porter's method for our dataset as demonstrated below.

Porter:
cmd: -it=USPTO-mdf-0.05 -cpc=G -emt | exec time: 07:23 secs

mobile device: 			    33.6833326760551
electronic device: 		    28.63492052752744
computing device: 		    25.539666723556127
display device: 		    23.69755247231993
compute device: 		    19.604581131580854
virtual machine: 		    16.725067554171893
user interface: 		    15.062028899069167
image form apparatus: 	    14.584135688497181
client device: 			    13.717931666935373
computer program product:   13.520757988739204
light source: 			    13.4761974473862
display panel: 			    12.987288891969184
unit configure: 		    11.988598669141473
display unit: 			    11.928201471077147
user device: 			    11.207295342544285
control unit: 			    10.304289943906731
mobile terminal: 		    8.968774302298257
far configure: 			    8.710208143729222
controller configure: 	    8.60326087325161
determine base: 		    8.435695146267795
touch panel: 			    8.340320405278447
optical fiber: 			    7.853598239644436

Curves:
cmd: -it=USPTO-mdf-0.05 -cpc=G -emt -cf | exec time: 07:48 secs

mobile device: 			    26.93560606060607
electronic device: 		    24.636363636363637
computing device: 		    20.659090909090924
display device: 		    19.962121212121207
compute device: 		    15.162878787878798
virtual machine: 		    14.348484848484855
optical fiber: 			    13.814393939393954
light source: 			    13.696969696969699
client device: 			    10.465909090909093
image form apparatus: 	    10.462121212121222
display unit: 			    10.272727272727273
unit configure: 		    10.151515151515154
user device: 			    9.503787878787884
display panel: 			    9.223484848484851
user interface: 		    8.833333333333329
touch panel: 			    7.844696969696972
control unit: 			    7.818181818181827
far configure: 			    7.393939393939394
computer storage medium: 	7.234848484848488
mobile terminal: 			6.91287878787879
controller configure: 		6.560606060606065
frequency band: 			6.3212121212121115

Again this method came with its own limitations especially when the timeseries plot had multiple curvatures.
### State space (Sonia)

## Prediction 2 IB

The popular terms are processed using either Porter or curves analysis to separate terms into
emerging (usage is increasing over time), stationary (usage is static) or declining (usage is
reduced over time). Note that we use the last 10 years with Porter's approach to label a term.

Given the labels, we take the top 25 emergent, top 25 stationary and top 25 declining terms
and run usage predictions on these terms.
The top emergent terms are defined as those with the most positive emergence score, the top stationary terms
those with a score around 0, and top declining those with the most negative score.

Different prediction techniques were implemented and tested, to determine the most suitable approach to predict future trends.
These techniques are now covered in the following sub-sections.

### Naive, linear, quadratic, cubic

A naive predictor used the last value in each time series as the predicted value for all future time instances. Linear, quadratic, or cubic predictors utilised linear, quadratic, or cubic functions fitted to each time series   to extrapolate future predicted values using those fitted parameters.

### ARIMA

**NOTE: Probably don't need ARIMA and Holt-Winters sub-section headers, e.g. after providing an initial list of techniques at the beginning of the prediction section.**

ARIMA (autoregressive integrated moving average) was applied using a grid search optimisation of its (p, d, q) parameters for each time series, based on training on the earliest 80% of the data and testing on the remaining 20% of data.  The grid search parameters were: p = [0, 1, 2, 4, 6], d = [0, 1, 2], q = [0, 1, 2].

### Holt-Winters

Holt-Winters was applied in its damped exponential smoothing form using an automated option for parameter optimisation for each time series. Holt-Winters' parameters include: alpha (smoothing level), beta (smoothing slope), and phi (damping slope).

### LSTM

Long Short-Term Memory (LSTM) recurrent neural networks are a powerful tool for detecting patterns in time series;
for predicting *n* values, three potential approaches are:

1. Single LSTM that can predict 1 value ahead (but is called *n* times on its own prediction to generate *n* values ahead)
2. Single LSTM that can predict *n* values ahead
3. *n* LSTM models, each model predicts different steps ahead (so merge all results to produce *n* values ahead)

The single LSTM with single lookahead can fail due to compound errors - once it goes wrong, its further predictions
are then based on erroneous output. A single LSTM predicting *n* outputs at once will have a single prediction pass and
in theory be less prone to compound error. Finally, multiple LSTMs each predicting a different step cannot suffer from
compound error as they are independent of each other.

In addition, we use Keras as our neural network library, where LSTMs can be trained as either stateless or stateful.
This means that when Keras trains the network, with a stateless LSTM, the LSTM state will not propagate between batches.
Conversely, with a stateful LSTM the state will propagate between batches. 

### Prediction Testing
**pyGrams** can be run in a testing mode, where the last *n* values are retained and not presented to the forecasting
algorithm - they are used to test its prediction. The residuals of the predictions are recorded and analysed; these
results are output as an HTML report. For example, using the supplied USPTO dataset:

```python pygrams.py -it USPTO-mdf-0.05 -emt --test -pns 0```

An extract of the output is shown below:

![img](img/prediction_test.png)

After examining the output, the predictors with lowest trimmed mean and standard deviation of 
relative root mean square error (of predicted vs actual) were found to be: naive, ARIMA, Holt-Winters,
stateful single LSTM with single look-ahead and stateful multiple LSTMs with single look-ahead.

### Results and Discussion

**pyGrams** was run on the example USPTO dataset of 3.2M patents, with predictions generated
from naive, ARIMA, Holt-Winters and stateful single LSTM with single look-ahead:

```python pygrams.py -it USPTO-mdf-0.05 -emt -pns 1 5 6 9```

Various outputs are produced; first, the top 250 popular terms are listed:
```
1. semiconductor device           3181.175539
2. electronic device              2974.360838
3. light source                   2861.643506
4. semiconductor substrate        2602.684013
5. mobile device                  2558.832724
6. pharmaceutical composition     2446.811441
7. electrically connect           2246.935926
8. base station                   2008.353328
9. memory cell                    1955.181403
10. display device                 1939.361315
...
```

Top emergent terms:
```
mobile device: 29.820764545328476
base station: 21.845790614296153
user equipment: 20.68596854844115
computer program product: 18.63254739799396
...
``` 

Top stationary terms:
```
...
fuel injection: 0.0004264722186599623
key performance indicator: 7.665313838841475e-05
chemical structure: -1.9375784177676214e-05
subsequent stage: -8.295764831296043e-05
...
```

Top declining terms:
```
...
plasma display panel: -5.010357686060525
optical disc: -5.487049355577173
semiconductor substrate: -6.777448387341435
liquid crystal display: -8.137031419798937
```

![img](img/prediction_test.png)



# Outputs 2 IT
## FDG
## Word cloud
## Graph summary

# Conclusion 1

# References
