import random

import pandas as pd
from tqdm import tqdm


class PatentsPickle2DataFrame(object):
    def __init__(self, data_frame_pickle_file_name, date_from=None, date_to=None, classification=None):

        self.__data_frame = pd.read_pickle(data_frame_pickle_file_name)
        self.__data_frame.reset_index(inplace=True, drop=True)

        if date_from is not None and date_to is not None:
            self.__subset_dates(date_from, date_to)

        if classification is not None:
            self._subset_cpc(classification)

    @property
    def data_frame(self):
        return self.__data_frame

    @data_frame.setter
    def data_frame(self, dfin):
        self.__data_frame = dfin

    def __subset_dates(self, date_from, date_to):
        _date_from = pd.Timestamp(date_from)
        _date_to = pd.Timestamp(date_to)

        self.__data_frame.sort_values('publication_date', inplace=True)
        print(f'Patents readied; {self.__data_frame.shape[0]:,} patents loaded')

        self.__data_frame.drop(self.__data_frame[(self.__data_frame.publication_date < _date_from) | (
                self.__data_frame.publication_date > _date_to)].index, inplace=True)
        print(f'{self.__data_frame.shape[0]:,} patents available after publication date sift')

        self.__data_frame.reset_index(inplace=True, drop=True)

    def _subset_cpc(self, cpc_in):
        indexes_to_delete = []
        for index, row in tqdm(self.__data_frame.iterrows(), desc='Sifting patents for ' + cpc_in, unit='patent',
                               total=self.__data_frame.shape[0]):
            cpc_list = row['classifications_cpc']
            cpc_found = False
            for cpc in cpc_list:
                if cpc.startswith(cpc_in):
                    cpc_found = True
            if not cpc_found:
                indexes_to_delete.append(index)

        self.__data_frame.drop(indexes_to_delete, inplace=True)
        self.__data_frame.reset_index(inplace=True, drop=True)

    def randomsample(self, random_seed, num_of_random_samples, in_date_order=False):
        if in_date_order:
            if random_seed:
                random.seed(a=random_seed)
            indices = random.sample(range(self.__data_frame.shape[0]), num_of_random_samples)
            sorted_indices = sorted(indices)
            return self.__data_frame[sorted_indices]
        else:
            if random_seed:
                return self.__data_frame.sample(num_of_random_samples, random_state=random_seed)
            else:
                return self.__data_frame.sample(num_of_random_samples)
