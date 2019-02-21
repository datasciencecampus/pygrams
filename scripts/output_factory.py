class OutputFactory(object):

    @staticmethod
    def get(output_type):
        if output_type == 'report':
            print(output_type)
        elif output_type == 'graph':
            print(output_type)
        elif output_type == 'wordcloud':
            print(output_type)
        elif output_type == 'term_counts_mat':
            print(output_type)
        else:
            assert 0, "Bad output type: " + type