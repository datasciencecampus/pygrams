import pandas as pd
class ArgsChecker:

    def __init__(self, args, args_default):
        self.args = args
        self.args_default = args_default

    def checkargs(self):
        app_exit = False

        if isinstance(self.args.year_to, str) & isinstance(self.args.year_from, str):
            if isinstance(self.args.month_to, str) & isinstance(self.args.month_from, str):
                if self.args.year_from + self.args.month_from > self.args.year_to + self.args.month_to:
                    print(f"year_to {self.args.year_to} and month_to {self.args.month_to} cannot be in the future "
                          f"of year_from {self.args.year_from} and month_from {self.args.month_from}")
                    app_exit = True
            else:
                if self.args.year_from > self.args.year_to:
                    print(f"year_to {self.args.year_to} cannot be in the future of year_from {self.args.year_from}")
                    app_exit = True
        else:
            if isinstance(self.args.month_to, str):
                if not isinstance(self.args.year_to, str):
                    print("year_to also needs to be defined to use month_to")
                    app_exit = True
            if isinstance(self.args.month_from, str):
                if not isinstance(self.args.year_from, str):
                    print("year_from also needs to be defined to use month_from")
                    app_exit = True

        if isinstance(self.args.year_from, str):
            if len(self.args.year_from) != 4:
                print(f"year_from {self.args.year_from} must be in YYYY format")
                app_exit = True

        if isinstance(self.args.month_from, str):
            if len(self.args.month_from) != 2:
                print(f"month_from {self.args.month_from} must be in MM format")
                app_exit = True

        if isinstance(self.args.year_to, str):
            if len(self.args.year_to) != 4:
                print(f"year_to {self.args.year_to} must be in YYYY format")
                app_exit = True

        if isinstance(self.args.month_to, str):
            if len(self.args.month_to) != 2:
                print(f"month_to {self.args.month_to} must be in MM format")
                app_exit = True

        if self.args.min_n > self.args.max_n:
            print(f"minimum ngram count {self.args.min_n} should be less or equal to maximum ngram "
                  f"count {self.args.max_n}")
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

        if app_exit:
            exit(0)

    def checkdf(self, df):
        app_exit = False

        if self.args.id_header is None:
            print(f"id_header not provided, will construct an id column")
            df.insert(0, 'id', range(0, 0 + len(df)))
            self.args.id_header = 'id'
            app_exit = False

        if self.args.id_header not in df.columns:
            print(f"id_header '{self.args.id_header}' not in dataframe")
            app_exit = True

        if self.args.text_header not in df.columns:
            print(f"text_header '{self.args.text_header}' not in dataframe")
            app_exit = True

        if isinstance(self.args.year_from, str):
            if self.args.date_header not in df.columns:
                print(f"date_header '{self.args.date_header}' not in dataframe")
                app_exit = True

        if isinstance(self.args.year_to, str):
            if self.args.date_header not in df.columns:
                print(f"date_header '{self.args.date_header}' not in dataframe")
                app_exit = True

        if 'termcounts' in self.args.output:
            if self.args.date_header not in df.columns:
                print(f"Cannot output termcounts without a specifying a date column")
                app_exit = True

        if app_exit:
            exit(0)

    def get_docs_mask_dict(self):

        year_to = pd.to_datetime('today').year if self.args.year_to is None else self.args.year_to
        month_to = pd.to_datetime('today').month if self.args.month_to is None else self.args.month_to

        docs_mask_dict = {}
        # doc_weights
        docs_mask_dict['filter_by'] = self.args.filter_by
        docs_mask_dict['cpc'] = [self.args.cpc_classification]
        docs_mask_dict['time'] = self.args.time
        docs_mask_dict['cite'] = []
        docs_mask_dict['columns'] = self.args.filter_columns
        docs_mask_dict['dates'] = [self.args.year_from, year_to, self.args.month_from, month_to, self.args.date_header]
        return docs_mask_dict

    def get_terms_mask_dict(self):
        terms_mask_dict = {}
        print()
