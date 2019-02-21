from tqdm import tqdm


class DocumentsFilter(object):
    def __init__(self, df, filter_columns, filter_by, cpc):
        print('processing doc filters')
        self.__doc_indices = set([])

        if filter_columns is not None:
            self.__doc_indices = self.__filter_column(df, filter_columns, filter_by)
        if cpc is not None:
            doc_set = self.__filter_cpc(df, cpc)
            if filter_by == 'intersection':
                self.__doc_indices = self.__doc_indices.intersection(set(doc_set))
            else:
                self.__doc_indices = self.__doc_indices.union(set(doc_set))


        # if filter_by == 'any':
        #     return filter_df[filter_df['filter'] > 0].index.values.tolist()
        # else:
        #     return filter_df[filter_df['filter'] == filter_df.shape[1] - 1].index.values.tolist()


    @property
    def doc_indices(self):
        return self.__doc_indices

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
