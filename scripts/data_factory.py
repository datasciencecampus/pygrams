import os

from pandas import read_pickle, read_excel, read_csv

from scripts.utils.pygrams_exception import PygramsException


def get(doc_source_file_name):

    if not os.path.isfile(doc_source_file_name):
        raise PygramsException('file: ' + doc_source_file_name + ' does not exist in data folder')

    if doc_source_file_name.endswith('.pkl.bz2') or doc_source_file_name.endswith('.pkl'):
        return read_pickle(doc_source_file_name)
    elif doc_source_file_name.endswith('.xls'):
        return read_excel(doc_source_file_name)
    elif doc_source_file_name.endswith('.csv'):
        return read_csv(doc_source_file_name, engine='python', error_bad_lines=False, skipinitialspace=True)
    elif doc_source_file_name.endswith('.xlsx'):
        return read_excel(doc_source_file_name)
    else:
        raise PygramsException('Unsupported file: ' + doc_source_file_name)
