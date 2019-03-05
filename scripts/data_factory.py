import pandas as pd
import os

from scripts.utils.pygrams_exception import PygramsException


def get(doc_source_file_name):

    if not os.path.isfile(doc_source_file_name):
        raise PygramsException('file: ' + doc_source_file_name + ' does not exist in data folder')

    if doc_source_file_name.endswith('.pkl.bz2'):
        return pd.read_pickle(doc_source_file_name)
    elif doc_source_file_name.endswith('.xls'):
        return pd.read_excel(doc_source_file_name)
    elif doc_source_file_name.endswith('.csv'):
        return pd.read_csv(doc_source_file_name, engine='python', error_bad_lines=False, skipinitialspace=True)
    elif doc_source_file_name.endswith('.xlsx'):
        return pd.read_excel(doc_source_file_name)
    else:
        raise PygramsException('Unsupported file: ' + doc_source_file_name)
