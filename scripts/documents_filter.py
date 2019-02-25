from tqdm import tqdm
import pandas as pd
from pandas import Timestamp
import calendar

class DocumentsFilter(object):
    def __init__(self, df, filter_columns, filter_by, cpc,
                 year_from=None, year_to=pd.to_datetime('today').year, month_from=None,
                 month_to=pd.to_datetime('today').month, dates_header=None):
        print('processing doc filters')
        self.__doc_indices = set([])

        if filter_columns is not None:
            self.__doc_indices = self.__filter_column(df, filter_columns, filter_by)
        if cpc is not None:
            doc_set = self.__filter_cpc(df, cpc)
            self.__add_set(doc_set, filter_by)

        if year_from is not None:
            doc_set = self.__filter_dates(df, year_from, year_to, month_from, month_to, dates_header)
            self.__add_set(doc_set, filter_by)

    def __add_set(self, doc_set, filter_by):
        if filter_by == 'intersection':
            if len(self.__doc_indices)>0:
                self.__doc_indices = self.__doc_indices.intersection(set(doc_set))
            else:
                self.__doc_indices = set(doc_set)
        else:
            self.__doc_indices = self.__doc_indices.union(set(doc_set))
    @property
    def doc_indices(self):
        return self.__doc_indices

    def __choose_last_day(self, year_in, month_in):
        return str(calendar.monthrange(int(year_in), int(month_in))[1])

    def __year2pandas_latest_date(self, year_in, month_in):
        if year_in is None:
            return Timestamp.now()

        if month_in is None:
            return Timestamp(str(year_in) + '-12-31')

        year_string = str(year_in) + '-' + str(month_in) + '-' + self.__choose_last_day(year_in, month_in)
        return Timestamp(year_string)

    def __year2pandas_earliest_date(self, year_in, month_in):
        if year_in is None:
            return Timestamp('2000-01-01')

        if month_in is None:
            return Timestamp(str(year_in) + '-01-01')

        year_string = str(year_in) + '-' + str(month_in) + '-01'

        return Timestamp(year_string)

    def __filter_cpc(self, df, cpc):
        cpc_index_list = []

        df = df.reset_index(drop=True)
        for index, row in tqdm(df.iterrows(), desc='Sifting documents for ' + cpc, unit='document',
                               total=df.shape[0]):
            cpc_list = row['classifications_cpc']
            for cpc_item in cpc_list:
                if cpc_item.startswith(cpc):
                    cpc_index_list.append(index)
                    break
        return cpc_index_list

    def __filter_column(self, df, filter_columns, filter_by):

        header_lists = []
        filter_headers = filter_columns.split(',')
        filter_headers = [header.strip() for header in filter_headers]
        header_filter_cols = [x.strip() for x in
                              filter_headers] if filter_columns is not None else []
        filter_df = df.copy()
        filter_df = filter_df[filter_headers]

        for column in filter_df:
            filter_df[column] = filter_df[column].replace({'No': 0, 'Yes': 1})
        filter_df['filter'] = filter_df.sum(axis=1)

        doc_set = None
        if len(header_filter_cols) > 0:
            for header_idx in range(len(header_filter_cols)):
                header_list = df[header_filter_cols[header_idx]]
                header_idx_list = []
                for row_idx, value in enumerate(header_list):
                    if value == 1 or value.lower() == 'yes':
                        header_idx_list.append(row_idx)
                header_lists.append(header_idx_list)
            doc_set = set(header_lists[0])

            for indices in header_lists[1:]:
                if filter_by == 'intersection':
                    doc_set = doc_set.intersection(set(indices))
                else:
                    doc_set =  doc_set.union(set(indices))
        return doc_set

    def __filter_dates(self, df, year_from, year_to, month_from, month_to, dates_header):
        date_from = self.__year2pandas_earliest_date(year_from, month_from)
        date_to = self.__year2pandas_latest_date(year_to, month_to)
        doc_ids=set([])

        date_from = pd.Timestamp(date_from)
        date_to = pd.Timestamp(date_to)

        for idx, date in enumerate(df[dates_header]):
            if date_to > date > date_from:
                doc_ids.add(idx)

        return doc_ids

