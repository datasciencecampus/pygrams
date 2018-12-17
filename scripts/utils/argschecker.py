class ArgsChecker():

    def __init__(self, args, args_default):
        self.__args = args
        self.__args_default= args_default

    def checkargs(self):
        app_exit = False
        if isinstance(self.__args.year_to,str) & isinstance(self.__args.year_from,str):
            if isinstance(self.__args.month_to, str) & isinstance(self.__args.month_from, str):
                if self.__args.year_from + self.__args.month_from > self.__args.year_to + args.month_to:
                    print(f"year_to {self.__args.year_to} and month_to {self.__args.month_to} cannot be in the future "
                          f"of year_from {self.__args.year_from} and month_from {self.__args.month_from}")
                    app_exit = True
            else:
                if self.__args.year_from > self.__args.year_to:
                    print(f"year_to {self.__args.year_to} cannot be in the future of year_from {self.__args.year_from}")
                    app_exit = True
        else:
            if isinstance(self.__args.month_to, str):
                if not isinstance(self.__args.year_to, str):
                    print("year_to also needs to be defined to use month_to")
                    app_exit = True
            if isinstance(self.__args.month_from, str):
                if not isinstance(self.__args.year_from, str):
                    print("year_from also needs to be defined to use month_from")
                    app_exit = True

        if isinstance(self.__args.year_from,str):
            if len(self.__args.year_from) != 4:
                print(f"year_from {self.__args.year_from} must be in YYYY format")
                app_exit = True

        if isinstance(self.__args.month_from, str):
            if len(self.__args.month_from) != 2:
                print(f"month_from {self.__args.month_from} must be in MM format")
                app_exit = True

        if isinstance(self.__args.year_to, str):
            if len(self.__args.year_to) != 4:
                print(f"year_to {self.__args.year_to} must be in YYYY format")
                app_exit = True

        if isinstance(self.__args.month_to, str):
            if len(self.__args.month_to) != 2:
                print(f"month_to {self.__args.month_to} must be in MM format")
                app_exit = True

        if self.__args.min_n > self.__args.max_n:
            print(f"minimum ngram count {self.__args.min_n} should be less or equal to maximum ngram "
                  f"count {self.__args.max_n}")
            app_exit = True

        if self.__args.num_ngrams_wordcloud < 20:
            print(f"at least 20 ngrams needed for wordcloud, {self.__args.num_ngrams_wordcloud} chosen")
            app_exit = True

        if self.__args.num_ngrams_report < 10:
            print(f"at least 10 ngrams needed for report, {args.num_ngrams_report} chosen")
            app_exit = True

        if self.__args.num_ngrams_fdg < 10:
            print(f"at least 10 ngrams needed for FDG, {self.__args.num_ngrams_fdg} chosen")
            app_exit = True

        if self.__args.focus_source != self.__args.focus_source:
            if self.__args.focus == None:
                print('argument [-fs] can only be used when focus is applied [-f]')
                app_exit = True

        if self.__args.output == 'table' or self.__args.output == 'all':
            if self.__args.focus == None:
                print('define a focus before requesting table (or all) output')
                app_exit = True

        if self.__args.wordcloud_title != self.__args_default.wordcloud_title or \
                self.__args.wordcloud_name != self.__args_default.wordcloud_name or \
                self.__args.num_ngrams_wordcloud != self.__args_default.num_ngrams_wordcloud:
                if args.output != 'wordcloud' or args.output != 'all':
                    print(args.wordcloud_title)
                    print('arguments [-wn] [-wt] [-nd] can only be used when output includes worldcloud '
                          '[-o] "wordcloud" or "all"')
                    app_exit = True

        if self.__args.report_name != self.__args_default.report_name or \
                self.__args.num_ngrams_report != self.__args_default.num_ngrams_report:
            if self.__args.output != 'report' or self.__args.output != 'all':
                print('arguments [-rn] [-np] can only be used when output includes report [-o] "report" or "all"')
                app_exit = True

        if self.__args.num_ngrams_fdg != self.__args_default.num_ngrams_fdg:
            if self.__args.output != 'fdg' or self.__args.output != 'all':
                print('argument [-nf] can only be used when output includes fdg [-o] "fdg" or "all]')
                app_exit = True

        if self.__args.table_name != self.__args_default.table_name:
            if self.__args.output != 'table' or self.__args.output != 'all':
                print('argument [-tn] can only be used when output includes table [-o] "table" or "all"')
                app_exit = True

        if app_exit:
            exit(0)