import argparse
import datetime
import os

import pandas as pd


def subsample_data_frame(input_pickle_name, subset_size, fraction, date_range, date_column_name):
    df = pd.read_pickle(input_pickle_name)

    if date_range is not None:
        df = df[(df[date_column_name] >= date_range[0]) & (df[date_column_name] <= date_range[1])]

    if fraction is not None:
        df = df.sample(frac=fraction)

    if subset_size is not None:
        df = df.sample(n=subset_size)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Reduces an existing pickled data frame by date or random subsampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # include defaults in help
    )

    parser.add_argument('-s', '--size', default=None, type=int,
                        help='Number of rows to store in pickle')
    parser.add_argument('-f', '--fraction', default=None, type=float,
                        help='Fraction of rows to store in pickle')

    parser.add_argument("-df", "--date-from", default=None,
                        help="The first date for the document cohort in YYYY/MM/DD format")
    parser.add_argument("-dt", "--date-to", default=None,
                        help="The last date for the document cohort in YYYY/MM/DD format")
    parser.add_argument("-dcn", "--date-column-name", default='publication_date',
                        help='Name of column containing date to filter by')

    parser.add_argument("input_pickle_name", metavar='<input pickle name>',
                        help="pickle containing data frame (.pkl.bz2 files)")
    parser.add_argument("output_folder_name", metavar='<output folder name>',
                        help="folder name where sub-sampled pickled data frame will be stored")

    args = parser.parse_args()
    if args.size and args.fraction:
        print('Cannot have both size and fraction defined')
        return

    date_from = None
    if isinstance(args.date_from, str):
        try:
            date_from = datetime.datetime.strptime(args.date_from, '%Y/%m/%d')
        except ValueError:
            raise ValueError(f"date_from defined as '{args.date_from}' which is not in YYYY/MM/DD format")

    date_to = None
    if isinstance(args.date_to, str):
        try:
            date_to = datetime.datetime.strptime(args.date_to, '%Y/%m/%d')
        except ValueError:
            raise ValueError(f"date_to defined as '{args.date_to}' which is not in YYYY/MM/DD format")

    if date_from is not None and date_to is not None:
        if date_from > date_to:
            raise ValueError(f"date_from '{args.date_from}' cannot be after date_to '{args.date_to}'")

    if date_from is not None and date_to is None:
        date_to = datetime.datetime.today()

    if date_from is None and date_to is not None:
        date_from = datetime.datetime(year=1900, month=1, day=1)

    date_range = None if date_from is None else [date_from, date_to]

    subset_suffix = '' if args.size is None else f'-S{args.size}'
    fraction_suffix = '' if args.fraction is None else f'-F{args.fraction}'
    date_suffix = '' if date_from is None else f'-{date_from:%Y-%m-%d}_{date_to:%Y-%m-%d}'

    if not args.input_pickle_name.endswith('.pkl.bz2'):
        print(f'Unhandled file extension on "{args.input_pickle_name}"; expected ".pkl.bz2"')
        return

    base_file_name = os.path.basename(args.input_pickle_name[:-8])

    pickle_file_name = os.path.join(args.output_folder_name,
                                    f'{base_file_name}{subset_suffix}{fraction_suffix}{date_suffix}.pkl.bz2')
    print(f'Storing sub-sampled data frame in pickle {pickle_file_name}')

    data_frame = subsample_data_frame(args.input_pickle_name,
                                      subset_size=args.size, fraction=args.fraction, date_range=date_range,
                                      date_column_name=args.date_column_name)

    print(f'After filtering: {data_frame.shape[0]} rows in data frame')
    print(f'Writing sub-sampled data frame in pickle {pickle_file_name}...')
    data_frame.to_pickle(pickle_file_name)
    print(f'...written sub-sampled data frame in pickle {pickle_file_name}')


if __name__ == '__main__':
    main()
