import datetime
from os import path

import pandas as pd

from pygrams.utils.date_utils import date_to_year_week
from pygrams.utils.pygrams_exception import PygramsException


class ArgsChecker:

    def __init__(self, args, args_default):
        self.args = args
        self.args_default = args_default

    def checkargs(self):
        app_exit = False

        doc_path = path.join(self.args.path, self.args.doc_source)
        if path.isfile(doc_path) is False:
            print(f"File {self.args.doc_source} in path {self.args.path} not found")
            app_exit = True

        date_from = None
        if isinstance(self.args.date_from, str):
            try:
                date_from = datetime.datetime.strptime(self.args.date_from, '%Y/%m/%d')
            except ValueError:
                raise PygramsException(f"date_from defined as '{self.args.date_from}' which is not in YYYY/MM/DD format")

        date_to = None
        if isinstance(self.args.date_to, str):
            try:
                date_to = datetime.datetime.strptime(self.args.date_to, '%Y/%m/%d')
            except ValueError:
                raise PygramsException(f"date_to defined as '{self.args.date_to}' which is not in YYYY/MM/DD format")

        if date_from is not None and date_to is not None:
            if date_from > date_to:
                raise PygramsException(f"date_from '{self.args.date_from}' cannot be after date_to '{self.args.date_to}'")

        ################

        timeseries_date_from = None
        if isinstance(self.args.timeseries_date_from, str):
            try:
                date_from = datetime.datetime.strptime(self.args.timeseries_date_from, '%Y/%m/%d')
            except ValueError:
                raise PygramsException(
                    f"date_from defined as '{self.args.date_from}' which is not in YYYY/MM/DD format")

        timeseries_date_to = None
        if isinstance(self.args.timeseries_date_to, str):
            try:
                timeseries_date_to = datetime.datetime.strptime(self.args.timeseries_date_to, '%Y/%m/%d')
            except ValueError:
                raise PygramsException(f"date_to defined as '{self.args.timeseries_date_to}' which is not in YYYY/MM/DD format")

        if timeseries_date_from is not None and timeseries_date_to is not None:
            if timeseries_date_from > timeseries_date_to:
                raise PygramsException(
                    f"date_from '{self.args.timeseries_date_from}' cannot be after date_to '{self.args.timeseries_date_to}'")
            if timeseries_date_from < date_from:
                raise PygramsException(
                    f"timeseries date_from '{self.args.timeseries_date_from}' cannot be before tfidf date_to '{self.args.date_from}'")
            if timeseries_date_to < date_to:
                raise PygramsException(
                    f"timeseries date_from '{self.args.timeseries_date_to}' cannot be aftere tfidf date_to '{self.args.date_to}'")

        ###############

        if len(self.args.search_terms) > 0:
            print("The user input words are:")
            for idx, word in enumerate(self.args.search_terms):
                print(f'{idx}. {word}')

        if self.args.use_cache is None and self.args.date_header is None:
            print()
            print('WARNING: No dates defined - time series analysis will not be possible with the cached object!')
            print()

        if self.args.min_ngrams > self.args.max_ngrams:
            print(f"minimum ngram count {self.args.min_ngrams} should be less or equal to maximum ngram "
                  f"count {self.args.max_ngrams}")
            app_exit = True

        if self.args.num_ngrams_wordcloud < 20:
            print(f"at least 20 ngrams needed for wordcloud, {self.args.num_ngrams_wordcloud} chosen")
            app_exit = True

        if self.args.num_ngrams_report < 10:
            print(f"at least 10 ngrams needed for report, {self.args.num_ngrams_report} chosen")
            app_exit = True

        if self.args.focus_source != self.args.focus_source:
            if self.args.focus is None:
                print('argument [-fs] can only be used when focus is applied [-f]')
                app_exit = True

        if 'table' in self.args.output:
            if self.args.focus is None:
                print('define a focus before requesting table (or all) output')
                app_exit = True

        if self.args.wordcloud_title != self.args_default.wordcloud_title or \
                self.args.num_ngrams_wordcloud != self.args_default.num_ngrams_wordcloud:
            if 'wordcloud' not in self.args.output:
                print(self.args.wordcloud_title)
                print('arguments [-wt] [-nd] can only be used when output includes wordcloud '
                      '[-o] "wordcloud"')
                app_exit = True

        if self.args.table_name != self.args_default.table_name:
            if 'table' not in self.args.output:
                print('argument [-tn] can only be used when output includes table [-o] "table"')
                app_exit = True

        if self.args.timeseries and self.args.use_cache is None and self.args.date_header is None:
            print(f"date_header is None")
            print(f"Cannot calculate emergence without a specifying a date column")
            app_exit = True

        if 'multiplot' in self.args.output and not self.args.timeseries:
            print("Cannot output multiplot without time series calculation")
            app_exit = True

        if app_exit:
            exit(0)

    def get_docs_mask_dict(self):
        docs_mask_dict = {'filter_by': self.args.filter_by,
                          'cpc': self.args.cpc_classification,
                          'cite': None, 'columns': self.args.filter_columns,
                          'date': None,
                          'timeseries_date': None,
                          'date_header': self.args.date_header
                          }

        if self.args.date_to is not None or self.args.date_from is not None:
            date_to = pd.to_datetime('today').date() if self.args.date_to is None else pd.to_datetime(self.args.date_to)
            date_from = pd.to_datetime('1900-01-01') if self.args.date_from is None else pd.to_datetime(
                self.args.date_from)
            docs_mask_dict['date'] = {
                'to': date_to_year_week(date_to),
                'from': date_to_year_week(date_from)
            }

        if self.args.timeseries_date_to is not None or self.args.timeseries_date_from is not None:
            timeseries_date_to = pd.to_datetime('today').date() if self.args.timeseries_date_to is None else pd.to_datetime(self.args.timeseries_date_to)
            timeseries_date_from = pd.to_datetime('1900-01-01') if self.args.timeseries_date_from is None else pd.to_datetime(self.args.timeseries_date_from)
            docs_mask_dict['timeseries_date'] = {
                'to': date_to_year_week(timeseries_date_to),
                'from': date_to_year_week(timeseries_date_from)
            }
        return docs_mask_dict

    def get_terms_mask_dict(self):
        terms_mask_dict = {}
        print()
