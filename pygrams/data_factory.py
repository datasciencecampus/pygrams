import os

from pandas import read_pickle, read_excel, read_csv

from pygrams.utils.pygrams_exception import PygramsException
from google.cloud import bigquery


def read_bq(doc_source_file_name):
    bqclient = bigquery.Client()

    # Download query results.
    query_string = """
    SELECT id, title, year, abstract, date_normal, date_online, date_print, journal 
    FROM `dimensions-ai.data_analytics.publications` 
    WHERE  not (abstract.preferred='null')  LIMIT 100000;
    """

    dataframe = (
        bqclient.query(query_string)
            .result()
            .to_dataframe(
            # Optionally, explicitly request to use the BigQuery Storage API. As of
            # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
            # API is used by default.
            create_bqstorage_client=True,
        )
    )
    return dataframe


def get(doc_source_file_name):

    if not os.path.isfile(doc_source_file_name) and not doc_source_file_name.endswith('.bq'):
        raise PygramsException('file: ' + doc_source_file_name + ' does not exist in data folder')

    if doc_source_file_name.endswith('.pkl.bz2') or doc_source_file_name.endswith('.pkl'):
        return read_pickle(doc_source_file_name)
    elif doc_source_file_name.endswith('.xls'):
        return read_excel(doc_source_file_name)
    elif doc_source_file_name.endswith('.csv'):
        return read_csv(doc_source_file_name, engine='python', error_bad_lines=False, skipinitialspace=True)
    elif doc_source_file_name.endswith('.xlsx'):
        return read_excel(doc_source_file_name)
    elif doc_source_file_name.endswith('.bq'):
        read_bq(doc_source_file_name)
    else:
        raise PygramsException('Unsupported file: ' + doc_source_file_name)
