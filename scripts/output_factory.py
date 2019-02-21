def get(output_type, output):
    if output_type == 'report':
        with open('outputs/reports/report.txt', 'w') as file:
            counter = 1
            for score, term in output:
                file.write(f' {term:30} {score:f}\n')
                print(f'{counter}. {term:30} {score:f}')
                counter+=1

        print(output_type)
    elif output_type == 'graph':
        print(output_type)
    elif output_type == 'wordcloud':
        print(output_type)
    elif output_type == 'term_counts_mat':
        print(output_type)
    else:
        assert 0, "Bad output type: " + output_type