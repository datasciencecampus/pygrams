import argparse
import datetime
import gc
import os
import time
from zipfile import ZipFile

import humanfriendly
import pandas as pd
import psutil
from numpy import int64, nan
from tqdm import tqdm


def __create_data_frame_from_patent_dates(_data_frame_pickle_folder_name, fraction, publication_date_range,
                                          is_file, zip_file_class, csv_to_df):
    # 'patent_id': no populated (needs other tables)
    # 'application_id': application_id => "appln_nr"
    # 'application_date': application_date => "appln_filing_date"
    # 'publication_date': publication_date => "earliest_publn_date"
    patstat_data_frame_list = []
    file_id = 1
    while is_file(os.path.join(_data_frame_pickle_folder_name, f'tls201_part{file_id:02}.zip')):
        zip_file_name = os.path.join(_data_frame_pickle_folder_name, f'tls201_part{file_id:02}.zip')

        with zip_file_class(zip_file_name) as zip_file:
            zip_file_contents = zip_file.namelist()

            for text_file_name in zip_file_contents:
                if text_file_name.endswith('.txt'):
                    # text_file_size = zip_file.getinfo(text_file_name).file_size
                    with zip_file.open(text_file_name) as table_data_file:
                        print(f'Reading "{text_file_name}"')
                        patstat_data_frame = csv_to_df(table_data_file)

                        # Consistent appln_id between tables...
                        patstat_data_frame.appln_id = patstat_data_frame.appln_id.astype(int64)

                        # 'application_id': application_id => "appln_nr_original"
                        patstat_data_frame.rename(columns={'appln_nr_original': 'application_id'}, inplace=True)

                        # 'application_date': application_date => "appln_filing_date"
                        patstat_data_frame.rename(columns={'appln_filing_date': 'application_date'}, inplace=True)

                        # 'publication_date': publication_date => "earliest_publn_date"
                        patstat_data_frame.rename(columns={'earliest_publn_date': 'publication_date'}, inplace=True)

                        patstat_data_frame.drop(
                            columns=['appln_filing_year', 'appln_kind', 'appln_nr', 'appln_nr_epodoc',
                                     'ipr_type', 'internat_appln_id', 'int_phase', 'reg_phase',
                                     'nat_phase',
                                     'earliest_filing_date', 'earliest_filing_year',
                                     'earliest_filing_id',
                                     'earliest_publn_year', 'earliest_pat_publn_id', 'granted',
                                     # 'docdb_family_id',
                                     'inpadoc_family_id', 'docdb_family_size', 'nb_citing_docdb_fam',
                                     'nb_applicants', 'nb_inventors'], inplace=True)

                        patstat_data_frame['application_date'] = pd.to_datetime(patstat_data_frame['application_date'],
                                                                                errors='coerce')
                        patstat_data_frame['publication_date'] = pd.to_datetime(patstat_data_frame['publication_date'],
                                                                                errors='coerce')

                        number_of_rows_pre_filter = patstat_data_frame.shape[0]
                        if publication_date_range is not None:
                            patstat_data_frame = patstat_data_frame[
                                (patstat_data_frame['publication_date'] >= publication_date_range[0])
                                & (patstat_data_frame['publication_date'] <= publication_date_range[1])]

                        if fraction is not None:
                            patstat_data_frame = patstat_data_frame.sample(frac=fraction)

                        number_of_rows_post_filter = patstat_data_frame.shape[0]

                        print(f'{number_of_rows_post_filter:,} rows out of {number_of_rows_pre_filter:,}'
                              ' remaining after filter')
                        clean_memory(text_file_name)

                        patstat_data_frame_list.append(patstat_data_frame)

        file_id += 1

    if file_id == 1:
        raise ValueError(f'Pickle base file name did not match any files; "{_data_frame_pickle_folder_name}"'
                         f' was post-fixed to generate "tls201_part00.zip')

    merged_data_frame = pd.concat(patstat_data_frame_list)
    del patstat_data_frame_list
    clean_memory('end of __create_data_frame_from_patent_dates')

    wierd_dates_df = merged_data_frame[merged_data_frame.publication_date.isnull()]
    print(f'Rejecting {wierd_dates_df.shape[0]:,} patents due to publication date'
          f' outside pandas max date of {pd.Timestamp.max}')
    merged_data_frame.drop(wierd_dates_df.index, inplace=True)

    merged_data_frame.set_index('appln_id', inplace=True, drop=False, verify_integrity=True)
    merged_data_frame = merged_data_frame.rename_axis(None)
    print(f'{merged_data_frame.shape[0]:,} total rows remaining after filter')
    return merged_data_frame


def __append_application_abstracts(_data_frame, _data_frame_pickle_folder_name,
                                   is_file, zip_file_class, csv_to_df):
    patstat_data_frame = pd.DataFrame()
    file_id = 1
    while is_file(os.path.join(_data_frame_pickle_folder_name, f'tls203_part{file_id:02}.zip')):
        zip_file_name = os.path.join(_data_frame_pickle_folder_name, f'tls203_part{file_id:02}.zip')

        with zip_file_class(zip_file_name) as zip_file:
            zip_file_contents = zip_file.namelist()

            for text_file_name in zip_file_contents:
                if text_file_name.endswith('.txt'):
                    # text_file_size = zip_file.getinfo(text_file_name).file_size
                    with zip_file.open(text_file_name) as table_data_file:
                        print(f'Reading "{text_file_name}"')
                        abstract_data_frame = csv_to_df(table_data_file)

                        abstract_data_frame.drop(columns=['appln_abstract_lg'], inplace=True)
                        abstract_data_frame.rename(columns={'appln_abstract': 'abstract'}, inplace=True)

                        abstract_data_frame.appln_id = abstract_data_frame.appln_id.astype(int64)

                        number_of_rows_pre_filter = abstract_data_frame.shape[0]
                        abstract_data_frame = _data_frame.merge(abstract_data_frame,
                                                                left_on='appln_id', right_on='appln_id',
                                                                how='inner', copy=False
                                                                )
                        number_of_rows_post_filter = abstract_data_frame.shape[0]

                        print(f'{number_of_rows_post_filter:,} rows out of {number_of_rows_pre_filter:,}'
                              ' remaining after merge with filtered publication data')

                        patstat_data_frame = pd.concat([patstat_data_frame, abstract_data_frame])

                        del abstract_data_frame
                        clean_memory(text_file_name)

        file_id += 1

    if file_id == 1:
        raise ValueError(f'Pickle base file name did not match any files; "{_data_frame_pickle_folder_name}"'
                         f' was post-fixed to generate "tls203_part00.zip')

    patstat_data_frame.set_index('appln_id', inplace=True, drop=False, verify_integrity=True)
    patstat_data_frame = patstat_data_frame.rename_axis(None)
    print(f'{patstat_data_frame.shape[0]:,} total rows remaining after filter')
    return patstat_data_frame


def __append_in_place_classifications_cpc(_data_frame, _data_frame_pickle_folder_name,
                                          is_file, zip_file_class, csv_to_df):
    # 'classifications_cpc': classifications_cpc => "cpc_class_symbol"

    _data_frame['classifications_cpc'] = None
    file_id = 1

    while is_file(os.path.join(_data_frame_pickle_folder_name, f'tls224_part{file_id:02}.zip')):
        zip_file_name = os.path.join(_data_frame_pickle_folder_name, f'tls224_part{file_id:02}.zip')

        with zip_file_class(zip_file_name) as zip_file:
            zip_file_contents = zip_file.namelist()

            for text_file_name in zip_file_contents:
                if text_file_name.endswith('.txt'):
                    # text_file_size = zip_file.getinfo(text_file_name).file_size
                    with zip_file.open(text_file_name) as table_data_file:
                        print(f'Reading "{text_file_name}"')
                        partial_patstat_data_frame = csv_to_df(table_data_file)

                        unknown_appln_ids = set()

                        # Consistent appln_id between tables...
                        partial_patstat_data_frame.appln_id = partial_patstat_data_frame.appln_id.astype(int64)

                        for row in tqdm(partial_patstat_data_frame.itertuples(),
                                        total=partial_patstat_data_frame.shape[0],
                                        desc=f'Processing {text_file_name}', unit='classification', unit_scale=True,
                                        leave=False):
                            appln_id = row.appln_id
                            cpc_code = row.cpc_class_symbol

                            try:
                                existing_cpc = _data_frame.at[appln_id, 'classifications_cpc']
                                if existing_cpc is not None:
                                    existing_cpc.append(cpc_code)
                                    _data_frame.at[appln_id, 'classifications_cpc'] = existing_cpc
                                else:
                                    _data_frame.at[appln_id, 'classifications_cpc'] = [cpc_code]

                            except KeyError:
                                unknown_appln_ids.add(appln_id)

                        print(
                            f'Processed {zip_file_name}; {len(unknown_appln_ids):,} out of '
                            f'{partial_patstat_data_frame.shape[0]:,} appln_id not in data_frame'
                            f' (due to family merging)')

        file_id += 1

    if file_id == 1:
        raise ValueError(f'Pickle base file name did not match any files; "{_data_frame_pickle_folder_name}"'
                         f' was post-fixed to generate "tls224_part00.zip')

    _data_frame.fillna(value=nan, inplace=True)


def __append_in_place_inventors(_data_frame, _data_frame_pickle_folder_name, is_file, zip_file_class, csv_to_df):
    # 'classifications_cpc': inventor_names => "cpc_class_symbol"

    patstat_data_frame_list = []
    file_id = 1

    print('Getting person data...')
    while is_file(os.path.join(_data_frame_pickle_folder_name, f'tls206_part{file_id:02}.zip')):
        zip_file_name = os.path.join(_data_frame_pickle_folder_name, f'tls206_part{file_id:02}.zip')

        with zip_file_class(zip_file_name) as zip_file:
            zip_file_contents = zip_file.namelist()

            for text_file_name in zip_file_contents:
                if text_file_name.endswith('.txt'):
                    # text_file_size = zip_file.getinfo(text_file_name).file_size
                    with zip_file.open(text_file_name) as table_data_file:
                        print(f'Reading "{text_file_name}"')
                        partial_patstat_data_frame = csv_to_df(table_data_file)

                        partial_patstat_data_frame.person_id = partial_patstat_data_frame.person_id.astype(int64)

                        partial_patstat_data_frame.drop(
                            columns=['person_name', 'doc_std_name_id', 'doc_std_name', 'psn_id',
                                     'psn_level'], inplace=True)

                        patstat_data_frame_list.append(partial_patstat_data_frame)

        file_id += 1

    if file_id == 1:
        raise ValueError(f'Pickle base file name did not match any files; "{_data_frame_pickle_folder_name}"'
                         f' was post-fixed to generate "tls206_part01.zip')

    patstat_person2names_data_frame = pd.concat(patstat_data_frame_list)
    del patstat_data_frame_list
    gc.collect()

    rows_before = patstat_person2names_data_frame.shape[0]
    patstat_person2names_data_frame.drop_duplicates(subset=['person_id'], keep='first', inplace=True)
    rows_after = patstat_person2names_data_frame.shape[0]
    print(f'Removed {rows_before - rows_after:,} duplicate rows (duplicate person_id)')

    patstat_person2names_data_frame.set_index('person_id', inplace=True, drop=False, verify_integrity=True)
    patstat_person2names_data_frame = patstat_person2names_data_frame.rename_axis(None)

    print('Getting map from application ID to person ID...')
    file_id = 1

    _data_frame.loc[:, 'applicant_organisation'] = None
    _data_frame.loc[:, 'applicant_countries'] = None
    _data_frame.loc[:, 'applicant_cities'] = None
    _data_frame.loc[:, 'inventor_names'] = None
    _data_frame.loc[:, 'inventor_countries'] = None
    _data_frame.loc[:, 'inventor_cities'] = None

    def append_to_entry(_appln_id, field_name, new_value):
        existing_field = _data_frame.at[_appln_id, field_name]
        if existing_field is not None:
            existing_field.append(new_value)
            _data_frame.at[_appln_id, field_name] = existing_field
        else:
            _data_frame.at[_appln_id, field_name] = [new_value]

    while is_file(os.path.join(_data_frame_pickle_folder_name, f'tls207_part{file_id:02}.zip')):
        zip_file_name = os.path.join(_data_frame_pickle_folder_name, f'tls207_part{file_id:02}.zip')

        with zip_file_class(zip_file_name) as zip_file:
            zip_file_contents = zip_file.namelist()

            for text_file_name in zip_file_contents:
                if text_file_name.endswith('.txt'):
                    # text_file_size = zip_file.getinfo(text_file_name).file_size
                    with zip_file.open(text_file_name) as table_data_file:
                        print(f'Reading "{text_file_name}"')
                        partial_patstat_data_frame = csv_to_df(table_data_file)

                        partial_patstat_data_frame.person_id = partial_patstat_data_frame.person_id.astype(int64)
                        partial_patstat_data_frame.appln_id = partial_patstat_data_frame.appln_id.astype(int64)
                        partial_patstat_data_frame.applt_seq_nr = partial_patstat_data_frame.applt_seq_nr.astype(int64)
                        partial_patstat_data_frame.invt_seq_nr = partial_patstat_data_frame.invt_seq_nr.astype(int64)

                        unknown_person_ids = set()
                        unknown_appln_ids = set()

                        for row in tqdm(partial_patstat_data_frame.itertuples(),
                                        total=partial_patstat_data_frame.shape[0],
                                        desc=f'Processing {text_file_name}', unit='file', unit_scale=True, leave=False):
                            appln_id = int64(row.appln_id)
                            person_id = int64(row.person_id)

                            try:
                                psn_row = patstat_person2names_data_frame.loc[person_id]

                                if pd.isna(psn_row.person_address):
                                    city = nan
                                else:
                                    city = psn_row.person_address.split(',')[-1].strip()

                                try:
                                    if row.applt_seq_nr > 0:
                                        append_to_entry(appln_id, 'applicant_organisation', psn_row.psn_name)
                                        append_to_entry(appln_id, 'applicant_countries', psn_row.person_ctry_code)
                                        append_to_entry(appln_id, 'applicant_cities', city)

                                    if row.invt_seq_nr > 0:
                                        append_to_entry(appln_id, 'inventor_names', psn_row.psn_name)
                                        append_to_entry(appln_id, 'inventor_countries', psn_row.person_ctry_code)
                                        append_to_entry(appln_id, 'inventor_cities', city)

                                except KeyError:
                                    unknown_appln_ids.add(appln_id)

                            except KeyError:
                                unknown_person_ids.add(person_id)

                        print(
                            f'Processed {zip_file_name}; {len(unknown_appln_ids):,} out of'
                            f' {partial_patstat_data_frame.shape[0]:,}'
                            f' appln_id not in data_frame (due to family merging),'
                            f' {len(unknown_person_ids)} unknown person_id entries found')

        file_id += 1

    if file_id == 1:
        raise ValueError(f'Pickle base file name did not match any files; "{_data_frame_pickle_folder_name}"'
                         f' was post-fixed to generate "tls207_part01.zip')


def __append_title(_data_frame, _data_frame_pickle_folder_name, is_file, zip_file_class, csv_to_df):
    # invention_title => "appln_title"
    patstat_data_frame_list = []
    file_id = 1

    while is_file(os.path.join(_data_frame_pickle_folder_name, f'tls202_part{file_id:02}.zip')):
        zip_file_name = os.path.join(_data_frame_pickle_folder_name, f'tls202_part{file_id:02}.zip')

        with zip_file_class(zip_file_name) as zip_file:
            zip_file_contents = zip_file.namelist()

            for text_file_name in zip_file_contents:
                if text_file_name.endswith('.txt'):
                    # text_file_size = zip_file.getinfo(text_file_name).file_size
                    with zip_file.open(text_file_name) as table_data_file:
                        print(f'Reading "{text_file_name}"')
                        partial_patstat_data_frame = csv_to_df(table_data_file)

                        partial_patstat_data_frame.appln_id = partial_patstat_data_frame.appln_id.astype(int64)

                        partial_patstat_data_frame.rename(columns={'appln_title': 'invention_title'}, inplace=True)

                        partial_patstat_data_frame.drop(columns=['appln_title_lg'], inplace=True)

                        patstat_data_frame_list.append(partial_patstat_data_frame)

        file_id += 1

    if file_id == 1:
        raise ValueError(f'Pickle base file name did not match any files; "{_data_frame_pickle_folder_name}"'
                         f' was post-fixed to generate "tls202_part01.zip')

    patstat_data_frame = pd.concat(patstat_data_frame_list)

    merged_data_frame = _data_frame.merge(patstat_data_frame,
                                          left_on='appln_id', right_on='appln_id',
                                          how='left', copy=False
                                          )

    merged_data_frame.set_index('appln_id', inplace=True, drop=False, verify_integrity=True)
    merged_data_frame = merged_data_frame.rename_axis(None)
    return merged_data_frame


def __remove_duplicates_by_family_id_and_use_as_patent_id(_data_frame):
    # Group by docdb_family_id
    # Sort by priority of appln_auth; GB, US, AU, NZ
    # Choose patent with largest abstract

    df_by_docdb_family_id = _data_frame.groupby('docdb_family_id')

    auth_score = {'GB': 4, 'US': 3, 'AU': 2, 'NZ': 1}

    with tqdm(total=_data_frame.shape[0], desc='Filtering by docdb_family_id', unit_scale=True, unit='row') as pbar:

        def none_or_sum(v):
            if v is None:
                return 0

            def len_with_nan(item):
                if pd.isna(item):
                    return 0
                return len(item)

            return sum(map(len_with_nan, v))

        def f(group):
            pbar.update(group.shape[0])

            # called twice on first group
            if group.shape[0] == 1:
                return group

            best_abstract_patent = None
            best_abstract_len = 0
            best_abstract_auth = 0

            best_applicant_patent = None
            best_applicant_cities_len = -1

            best_inventor_patent = None
            best_inventor_cities_len = -1

            for row in group.itertuples(index=False):

                current_applicant_cities_len = none_or_sum(row.applicant_cities)
                if best_applicant_patent is None or current_applicant_cities_len > best_applicant_cities_len:
                    best_applicant_cities_len = current_applicant_cities_len
                    best_applicant_patent = row

                current_inventor_cities_len = none_or_sum(row.inventor_cities)
                if best_inventor_patent is None or current_inventor_cities_len > best_inventor_cities_len:
                    best_inventor_cities_len = current_inventor_cities_len
                    best_inventor_patent = row

                if not isinstance(row.abstract, str):
                    print()
                    print(f'Non-string abstract for patent appln_id={row.appln_id}, abstract="{row.abstract}"')
                    continue

                if best_abstract_patent is None:
                    best_abstract_auth = auth_score.get(row.appln_auth, 0)
                    best_abstract_patent = row
                    best_abstract_len = len(row.abstract)
                else:
                    current_auth = auth_score.get(row.appln_auth, 0)
                    if current_auth > best_abstract_auth:
                        best_abstract_auth = current_auth
                        best_abstract_patent = row
                        best_abstract_len = len(row.abstract)
                    elif current_auth == best_abstract_auth:
                        current_abstract_len = len(row.abstract)
                        if current_abstract_len > best_abstract_len:
                            best_abstract_auth = current_auth
                            best_abstract_patent = row
                            best_abstract_len = current_abstract_len

            df = pd.DataFrame([best_abstract_patent])
            df.at[0, 'applicant_organisation'] = best_applicant_patent.applicant_organisation
            df.at[0, 'applicant_countries'] = best_applicant_patent.applicant_countries
            df.at[0, 'applicant_cities'] = best_applicant_patent.applicant_cities
            df.at[0, 'inventor_names'] = best_inventor_patent.inventor_names
            df.at[0, 'inventor_countries'] = best_inventor_patent.inventor_countries
            df.at[0, 'inventor_cities'] = best_inventor_patent.inventor_cities

            return df

        start_time = time.time()
        filtered_df = df_by_docdb_family_id.apply(f)
        elapsed_time_secs = time.time() - start_time
        m, s = divmod(elapsed_time_secs, 60)
        h, m = divmod(m, 60)
        print(f'Family id filtering took {h:.0f}:{m:02.0f}:{s:02.1f} ({elapsed_time_secs:,.1f}s)')

    filtered_df.rename(columns={'docdb_family_id': 'patent_id'}, inplace=True)

    filtered_df.set_index('appln_id', inplace=True, drop=False, verify_integrity=True)
    filtered_df = filtered_df.rename_axis(None)

    filtered_df.fillna(value=nan, inplace=True)

    null_counts = filtered_df.isnull().sum()
    print(
        f'{null_counts.applicant_cities:,} patents with null applicant cities out of {filtered_df.shape[0]:,} patents')
    print(f'{null_counts.inventor_cities:,} patents with null inventor cities out of {filtered_df.shape[0]:,} patents')

    return filtered_df


def __lite_remove_duplicates_by_family_id_and_use_as_patent_id(_data_frame):
    _data_frame.drop(columns=['application_id', 'application_date', 'appln_auth', 'receiving_office'], inplace=True)

    print('Recording length of abstracts...')
    _data_frame['len'] = _data_frame['abstract'].str.len()

    print('Grouping by family id...')
    df_grouped = _data_frame.groupby(['docdb_family_id']).agg({'len': 'max'})
    df_grouped = df_grouped.reset_index()
    df_grouped = df_grouped.rename(columns={'len': 'len_max'})

    # appln_id 2, 76, 118 are duped

    print('Merging data frame with max length abstract per family id...')
    _data_frame = pd.merge(_data_frame, df_grouped, how='left', on=['docdb_family_id'])

    print('Subsetting data frame to leave max length abstract per family id...')
    _data_frame = _data_frame[_data_frame['len'] == _data_frame['len_max']]
    _data_frame.drop(columns=['len', 'len_max'], inplace=True)
    _data_frame.rename(columns={'docdb_family_id': 'patent_id'}, inplace=True)

    _data_frame.drop_duplicates(subset=['patent_id'], keep='first', inplace=True)

    _data_frame.set_index('appln_id', inplace=True, drop=False, verify_integrity=True)
    _data_frame = _data_frame.rename_axis(None)
    return _data_frame


def clean_memory(stage_name):
    process = psutil.Process(os.getpid())
    bytes_in_use_before_gc = process.memory_info().rss

    gc.collect()

    bytes_in_use_after_gc = process.memory_info().rss

    bytes_released = bytes_in_use_before_gc - bytes_in_use_after_gc

    print('After ' + stage_name + ': ' + humanfriendly.format_size(bytes_in_use_after_gc) + ' in use ('
          + humanfriendly.format_size(bytes_released) + ' released)')


def __subset_data_frame(_data_frame, subset_size):
    return _data_frame.sample(n=subset_size)


def read_df_from_pickle(pickle_file_name, read_pickle):
    print(f'Loading {pickle_file_name}...')
    df = read_pickle(pickle_file_name)
    print(f'...loaded {pickle_file_name}')
    return df


def cache_state(__data_frame, pickle_file_name, message, to_pickle):
    print(f'Pickling {__data_frame.shape[0]:,} patents to {pickle_file_name}...')
    to_pickle(__data_frame, pickle_file_name)
    print(f'...pickled to {pickle_file_name}')
    clean_memory(message)
    print()


def convert_patstat_data_to_data_frame(data_frame_pickle_folder_name, output_folder_name, lite_data=False,
                                       subset_size=None, fraction=None, date_range=None,
                                       is_file=os.path.isfile, zip_file_class=ZipFile, csv_to_df=pd.read_csv,
                                       read_pickle=pd.read_pickle, to_pickle=pd.to_pickle):
    start_time = time.time()

    full_suffix = 'lite-' if lite_data else 'full-'
    subset_suffix = '' if subset_size is None else f'S{subset_size}-'
    fraction_suffix = '' if fraction is None else f'F{fraction}-'
    date_suffix = '' if date_range is None else f'{date_range[0]:%Y-%m-%d}_{date_range[1]:%Y-%m-%d}-'

    base_pickle_name = os.path.join(output_folder_name,
                                    f'df-{full_suffix}{subset_suffix}{fraction_suffix}{date_suffix}')

    print()

    # 'patent_id': patent_id,
    # 'application_id': appln_nr,
    # 'application_date': application_date,
    # 'publication_date': publication_date,
    publication_pickle_name = base_pickle_name + '1-publication.pkl.bz2'
    if not is_file(publication_pickle_name):
        __data_frame = __create_data_frame_from_patent_dates(data_frame_pickle_folder_name, fraction, date_range,
                                                             is_file, zip_file_class, csv_to_df)
        if subset_size is not None:
            __data_frame = __subset_data_frame(__data_frame, subset_size)
            clean_memory('1.9-subset publications')

        cache_state(__data_frame, publication_pickle_name, '1-publication', to_pickle)

    # 'appln_id': key field to bind tables
    # 'abstract': abstract
    abstract_pickle_name = base_pickle_name + '2-abstract.pkl.bz2'
    if not is_file(abstract_pickle_name):
        __data_frame = read_df_from_pickle(publication_pickle_name, read_pickle)
        __data_frame = __append_application_abstracts(__data_frame, data_frame_pickle_folder_name,
                                                      is_file, zip_file_class, csv_to_df)
        cache_state(__data_frame, abstract_pickle_name, '2-abstracts', to_pickle)

    if lite_data:
        dedupe_pickle_name = base_pickle_name + '3-lite-grouped-by.pkl.bz2'
        if not is_file(dedupe_pickle_name):
            __data_frame = read_df_from_pickle(abstract_pickle_name, read_pickle)
            __data_frame = __lite_remove_duplicates_by_family_id_and_use_as_patent_id(__data_frame)
            cache_state(__data_frame, dedupe_pickle_name, '3-lite-grouped-by', to_pickle)

        classifications_pickle_name = base_pickle_name + '4-lite-classifications.pkl.bz2'
        if not is_file(classifications_pickle_name):
            __data_frame = read_df_from_pickle(dedupe_pickle_name, read_pickle)
            __append_in_place_classifications_cpc(__data_frame, data_frame_pickle_folder_name,
                                                  is_file, zip_file_class, csv_to_df)
            cache_state(__data_frame, classifications_pickle_name, '4-lite-classifications', to_pickle)

        __data_frame = read_df_from_pickle(classifications_pickle_name, read_pickle)

    else:
        # 'psn_name': [inventor_names, applicant_orgnames],
        # 'person_ctry_code': [inventor_countries, applicant_countries],
        # 'person_address': [inventor_cities, applicant_cities],
        person_pickle_name = base_pickle_name + '3-person.pkl.bz2'
        if not is_file(person_pickle_name):
            __data_frame = read_df_from_pickle(abstract_pickle_name, read_pickle)
            __append_in_place_inventors(__data_frame, data_frame_pickle_folder_name,
                                        is_file, zip_file_class, csv_to_df)
            cache_state(__data_frame, person_pickle_name, '3-person', to_pickle)

        # 'invention_title': invention_title,
        title_pickle_name = base_pickle_name + '4-title.pkl.bz2'
        if not is_file(title_pickle_name):
            __data_frame = read_df_from_pickle(person_pickle_name, read_pickle)
            __data_frame = __append_title(__data_frame, data_frame_pickle_folder_name,
                                          is_file, zip_file_class, csv_to_df)
            cache_state(__data_frame, title_pickle_name, '4-title', to_pickle)

        # Remove duplicates for family id
        dedupe_pickle_name = base_pickle_name + '5-dedupe.pkl.bz2'
        if not is_file(dedupe_pickle_name):
            __data_frame = read_df_from_pickle(title_pickle_name, read_pickle)
            __data_frame = __remove_duplicates_by_family_id_and_use_as_patent_id(__data_frame)
            cache_state(__data_frame, dedupe_pickle_name, '5-dedupe', to_pickle)

        # 'classifications_cpc': [classifications_cpc],
        classifications_pickle_name = base_pickle_name + '6-classifications.pkl.bz2'
        if not is_file(classifications_pickle_name):
            __data_frame = read_df_from_pickle(dedupe_pickle_name, read_pickle)
            __append_in_place_classifications_cpc(__data_frame, data_frame_pickle_folder_name,
                                                  is_file, zip_file_class, csv_to_df)
            cache_state(__data_frame, classifications_pickle_name, '6-classifications', to_pickle)

        __data_frame = read_df_from_pickle(classifications_pickle_name, read_pickle)

        # Yet to populate:
        #
        # 'related_document_ids': [related_document_ids],
        # 'claim1': claim1,  # in the future we can consider that being a list with all claims
        # 'patents_cited': [cited_patents]

    __data_frame.drop(columns=['appln_id'], inplace=True)
    clean_memory('7-dropped appln_id')

    __data_frame.set_index('patent_id', inplace=True, drop=False, verify_integrity=True)
    __data_frame.sort_values('publication_date', inplace=True)

    elapsed_time_secs = time.time() - start_time
    m, s = divmod(elapsed_time_secs, 60)
    h, m = divmod(m, 60)
    print(f'IPO patents readied; {__data_frame.shape[0]:,} patents loaded in'
          f' {h:.0f}:{m:02.0f}:{s:02.1f} ({elapsed_time_secs:,.1f}s)')

    return __data_frame


def main():
    parser = argparse.ArgumentParser(description="Produces pickle data frame from patstat data; pickle will be named "
                                                 "patstat-full.pkl.bz2")

    parser.add_argument('-l', '--lite', action='store_true',
                        help='Stores minimum columns in data frame as required by pgrams')
    parser.add_argument('-s', '--size', default=None, type=int,
                        help='Number of patents to store in pickle')
    parser.add_argument('-f', '--fraction', default=None, type=float,
                        help='Fraction of patents to store in pickle')

    parser.add_argument("-df", "--date-from", default=None,
                        help="The first date for the document cohort in YYYY/MM/DD format")
    parser.add_argument("-dt", "--date-to", default=None,
                        help="The last date for the document cohort in YYYY/MM/DD format")

    parser.add_argument("input_folder_name", metavar='<input folder name>',
                        help="folder containing patstat database export files (.zip files)")
    parser.add_argument("output_folder_name", metavar='<output folder name>',
                        help="folder name where pickled data frames will be stored")

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

    full_suffix = 'lite' if args.lite else 'full'
    subset_suffix = '' if args.size is None else f'-S{args.size}'
    fraction_suffix = '' if args.fraction is None else f'-F{args.fraction}'
    date_suffix = '' if date_from is None else f'-{date_from:%Y-%m-%d}_{date_to:%Y-%m-%d}'

    pickle_file_name = os.path.join(args.output_folder_name,
                                    f'patstat-{full_suffix}{subset_suffix}{fraction_suffix}{date_suffix}.pkl.bz2')
    print(f'Storing patstat patents in pickle {pickle_file_name}')

    data_frame = convert_patstat_data_to_data_frame(args.input_folder_name, args.output_folder_name,
                                                    lite_data=args.lite, subset_size=args.size,
                                                    fraction=args.fraction, date_range=date_range)

    print(f'Writing patstat patents in pickle {pickle_file_name}...')
    data_frame.to_pickle(pickle_file_name)
    print(f'Writing patstat patents in pickle {pickle_file_name}... done')


if __name__ == '__main__':
    main()
