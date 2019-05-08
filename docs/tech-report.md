# Patent report 12-15 A4 pages


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
## Previous and related work
## Filtering 2
### CPC
### Stop words
#### Manual list
#### Fatima work
### Word embedding 1 E
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
