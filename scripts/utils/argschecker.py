import datetime
from os import path
import pandas as pd

from scripts.utils.pygrams_exception import PygramsException


class ArgsChecker:

    def __init__(self, args, args_default):
        self.args = args
        self.args_default = args_default

    def checkargs(self):
        app_exit = False

        if path.isfile(path.join(self.args.path, self.args.doc_source)) is False:
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

        if self.args.num_ngrams_fdg < 10:
            print(f"at least 10 ngrams needed for FDG, {self.args.num_ngrams_fdg} chosen")
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

        if self.args.num_ngrams_report != self.args_default.num_ngrams_report:
            if 'report' not in self.args.output:
                print('arguments [-np] can only be used when output includes report [-o] "report"')
                app_exit = True

        if self.args.num_ngrams_fdg != self.args_default.num_ngrams_fdg:
            if 'fdg' not in self.args.output:
                print('argument [-nf] can only be used when output includes fdg [-o] "fdg"')
                app_exit = True

        if self.args.table_name != self.args_default.table_name:
            if 'table' not in self.args.output:
                print('argument [-tn] can only be used when output includes table [-o] "table"')
                app_exit = True

        invalid_predictor_names = []
        for i in self.args.predictor_names:
            if i >= 13:
                invalid_predictor_names.append(i)

        if len(invalid_predictor_names) > 0:
            print(f"invalid predictor name number(s) {' '.join(str(e) for e in invalid_predictor_names)} provided (must be between 0 and 12)")
            app_exit = True

        if self.args.emerging_technology:
            if self.args.date_header is None:
                print(f"date_header is None")
                print(f"Cannot calculate emergence without a specifying a date column")
                app_exit = True

        if app_exit:
            exit(0)

    def get_docs_mask_dict(self):
        docs_mask_dict = {'filter_by': self.args.filter_by,
                          'cpc': self.args.cpc_classification,
                          'time': self.args.time,
                          'cite': None, 'columns': self.args.filter_columns,
                          'date': None,
                          'date_header': self.args.date_header
                          }

        if self.args.date_to is not None and self.args.date_from is not None:
            date_to = pd.to_datetime('today').date() if self.args.date_to is None else pd.to_datetime(self.args.date_to)
            date_from = pd.to_datetime('1900-01-01') if self.args.date_from is None else pd.to_datetime(
                self.args.date_from)
            docs_mask_dict['date'] = {
                'to': date_to,
                'from': date_from
            }
        return docs_mask_dict

    def get_terms_mask_dict(self):
        terms_mask_dict = {}
        print()
