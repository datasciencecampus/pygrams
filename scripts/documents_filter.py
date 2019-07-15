from tqdm import tqdm


class DocumentsFilter(object):
    def __init__(self, dates, docs_mask_dict, cpc_dict, number_of_docs):
        self.__doc_indices = set([])
        self.__cpc_dict = cpc_dict

        # todo: Support or fully remove column filtering
        # if docs_mask_dict['columns'] is not None:
        #     self.__doc_indices = self.__filter_column(dates, docs_mask_dict['columns'], docs_mask_dict['filter_by'])

        if docs_mask_dict['cpc'] is not None:
            doc_set = self.__filter_cpc(docs_mask_dict['cpc'])
            self.__add_set(doc_set, docs_mask_dict['filter_by'])

        if docs_mask_dict['date'] is not None:
            doc_set = self.__filter_dates(dates, docs_mask_dict['date'])
            self.__add_set(doc_set, docs_mask_dict['filter_by'])

        self.__doc_filters = [0.0] * number_of_docs if len(self.__doc_indices) > 0 else [1.0] * number_of_docs
        for i in self.__doc_indices:
            self.__doc_filters[i] = 1.0

    def __add_set(self, doc_set, filter_by):
        if filter_by == 'intersection':
            if len(self.__doc_indices) > 0:
                self.__doc_indices = self.__doc_indices.intersection(set(doc_set))
            else:
                self.__doc_indices = set(doc_set)
        else:
            self.__doc_indices = self.__doc_indices.union(set(doc_set))

    @property
    def doc_filters(self):
        return self.__doc_filters

    @property
    def doc_indices(self):
        return self.__doc_indices

    def __filter_cpc(self, cpc):
        indices_set = set()
        for cpc_item in tqdm(self.__cpc_dict, desc='Sifting documents for cpc class: ' +
                             str(cpc), unit='document', total=len(self.__cpc_dict)):
            if cpc_item.startswith(cpc):
                indices_set |= self.__cpc_dict[cpc_item]

        return list(indices_set)

    @staticmethod
    def __filter_column(dates, filter_columns, filter_by):

        header_lists = []
        filter_headers = filter_columns.split(',')
        filter_headers = [header.strip() for header in filter_headers]
        header_filter_cols = [x.strip() for x in
                              filter_headers] if filter_columns is not None else []
        filter_df = dates.copy()
        filter_df = filter_df[filter_headers]

        for column in filter_df:
            filter_df[column] = filter_df[column].replace({'No': 0, 'Yes': 1})
        filter_df['filter'] = filter_df.sum(axis=1)

        doc_set = None
        if len(header_filter_cols) > 0:
            for header_idx in range(len(header_filter_cols)):
                header_list = dates[header_filter_cols[header_idx]]
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
                    doc_set = doc_set.union(set(indices))
        return doc_set

    @staticmethod
    def __filter_dates(dates, date_dict):
        date_from = date_dict['from']
        date_to = date_dict['to']
        doc_ids = set([])

        for idx, date in tqdm(enumerate(dates), desc='Sifting documents for date-range: ' +
                                                     str(date_from) + ' - ' + str(date_to),
                              unit='document',
                              total=dates.shape[0]):
            if date_to > date > date_from:
                doc_ids.add(idx)

        return doc_ids
