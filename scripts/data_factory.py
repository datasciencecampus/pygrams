import pandas as pd


def get(doc_source_file_name):
    if doc_source_file_name.endswith('.pkl.bz2'):
        return pd.read_pickle(doc_source_file_name)
    elif doc_source_file_name.endswith('.xls'):
        return pd.read_excel(doc_source_file_name)
    elif doc_source_file_name.endswith('.csv'):
        return pd.read_csv(doc_source_file_name, engine='python', error_bad_lines=False)
    elif doc_source_file_name.endswith('.xlsx'):
        return pd.read_excel(doc_source_file_name)
