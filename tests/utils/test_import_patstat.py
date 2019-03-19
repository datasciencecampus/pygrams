import math
import os
import unittest

import numpy as np
import pandas as pd

from scripts.utils.import_patstat import convert_patstat_data_to_data_frame


def create_df_with_unused_columns(df_dict, unused_keys):
    dict_key = list(df_dict.keys())[0]
    num_copies = len(df_dict[dict_key])
    for unused_key in unused_keys:
        df_dict[unused_key] = [unused_key] * num_copies
    return pd.DataFrame(df_dict)


class TestPatstatToDataFrame(unittest.TestCase):
    df = None  # will be populated by setUpClass

    # noinspection PyBroadException
    def assertIsNaN(self, value):
        msg = f'{str(value)} is not NaN'
        try:
            if not math.isnan(value):
                self.fail(msg)
        except:
            self.fail(msg)

    def setup(self):
        # Design of test data:
        # appln_id defines patent UID
        # family id 'family45' contains appln_id 99, 01, 02
        # family id 'family123' contains appln_id 09
        # In family45:
        #   appln_id 99 has no associated applicants or inventors; DE abstract
        #   appln_id 01 has 2 inventors (12,21)[best cities], 1 applicant (20) and AU abstract
        #   appln_id 02 has 2 inventors (76,1), 1 applicant (8) [best city] and GB abstract [best abstract]
        long_abstract1_same_length = 'abstract for patent appln_id 01 longest here'
        long_abstract2_same_length = 'abstract for patent appln_id 02 longest blah'
        df_abstract0 = pd.DataFrame({
            'appln_id': ['09', '01'],
            'appln_abstract_lg': ['abslg0', 'abslg1'],
            'appln_abstract': ['abstract for patent appln_id 09', long_abstract1_same_length],
        })
        df_abstract1 = pd.DataFrame({
            'appln_id': ['02', '99', '0'],
            'appln_abstract_lg': ['abslg2', 'abslg99', 'abslg0'],
            'appln_abstract': [long_abstract2_same_length, 'abstract for appln_id 99',
                               'abstract for patent appln_id 0'],
        })

        title_unused_columns = ['appln_title_lg']
        df_title0 = create_df_with_unused_columns({
            'appln_id': ['09', '01'],
            'appln_title': ['Title of patent #09', "Patent #01's title"]},
            title_unused_columns)
        df_title1 = create_df_with_unused_columns({
            'appln_id': ['02'],
            'appln_title': ['Title 2']},
            title_unused_columns)

        applications_unused_columns = ['appln_kind', 'appln_nr', 'appln_nr_epodoc', 'ipr_type', 'internat_appln_id',
                                       'int_phase', 'reg_phase', 'nat_phase', 'earliest_filing_date',
                                       'earliest_filing_year', 'earliest_filing_id', 'earliest_publn_year',
                                       'earliest_pat_publn_id', 'granted', 'inpadoc_family_id', 'docdb_family_size',
                                       'nb_citing_docdb_fam', 'nb_applicants', 'nb_inventors', 'receiving_office']
        df_applications0 = create_df_with_unused_columns({
            'appln_id': ['99', '09', '01'],
            'appln_nr_original': ['apporig_99', 'apporig_09', 'apporig_01'],
            'appln_filing_date': ['1999-01-01', '2010-10-01', '1999-07-12'],
            'earliest_publn_date': ['2001-07-21', '2011-02-12', '1999-11-23'],
            'appln_filing_year': ['1999', '2010', '1999'],
            'appln_auth': ['DE', 'US', 'AU'],
            'docdb_family_id': ['family45', 'family123', 'family45']},
            applications_unused_columns)
        df_applications1 = create_df_with_unused_columns({
            'appln_id': ['02', '0'],
            'appln_nr_original': ['app_orig02', 'app_orig0'],
            'appln_filing_date': ['2007-02-28', '1999-10-17'],
            'earliest_publn_date': ['2007-07-08', '1999-11-21'],
            'appln_filing_year': ['2007', '1999'],
            'appln_auth': ['GB', 'RU'],
            'docdb_family_id': ['family45', 'family0']},
            applications_unused_columns)

        cpc_unused_columns = ['cpc_scheme', 'cpc_version', 'cpc_value', 'cpc_position', 'cpc_gener_auth']
        df_cpc0 = create_df_with_unused_columns({
            'appln_id': ['09', '01'],
            'cpc_class_symbol': ['Q99Q 123/456', 'Y02L 238/7209']},
            cpc_unused_columns)
        df_cpc1 = create_df_with_unused_columns({
            'appln_id': ['02', '02'],
            'cpc_class_symbol': ['H01L  24/03', 'Y02L2224/85203']},
            cpc_unused_columns)

        # APPLT_SEQ_NR >0 => applicant
        # INVT_SEQ_NR >0 => inventor
        personapp_unused_columns = []
        df_personapp0 = create_df_with_unused_columns({
            'person_id': ['1', '2', '76', '12'],
            'appln_id': ['09', '09', '02', '01'],
            'applt_seq_nr': ['0', '1', '0', '0'],
            'invt_seq_nr': ['1', '0', '1', '1']},
            personapp_unused_columns)
        df_personapp1 = create_df_with_unused_columns({
            'person_id': ['8', '1', '20', '21'],
            'appln_id': ['02', '02', '01', '01'],
            'applt_seq_nr': ['1', '0', '1', '0'],
            'invt_seq_nr': ['0', '2', '0', '2']},
            personapp_unused_columns)

        person_unused_columns = ['person_name', 'doc_std_name_id', 'doc_std_name', 'psn_id', 'psn_level']
        df_person0 = create_df_with_unused_columns({
            'person_id': ['1', '2', '12'],
            'person_address': ['73527 Schwäbisch Gmünd', 'somewhere else', np.NaN],
            'person_ctry_code': ['DE', 'AU', 'US'],
            'psn_name': ['JONES FRED', 'SMITH INDUSTRIES', 'A N OTHER'],
            'psn_sector': ['INDIVIDUAL', 'COMPANY', 'INDIVIDUAL']},
            person_unused_columns)
        df_person1 = create_df_with_unused_columns({
            'person_id': ['76', '8'],
            'person_address': ['Ontario N2L 3W8', 'City Road, Newport NP20 1XJ'],
            'person_ctry_code': ['US', 'GB'],
            'psn_name': ['A N OTHER', 'FISH I AM'],
            'psn_sector': ['INDIVIDUAL', 'COMPANY']},
            person_unused_columns)
        df_person2 = create_df_with_unused_columns({
            'person_id': ['20', '21'],
            'person_address': ['home', 'Richard-Bullinger-Strasse 77, 73527 Schwäbisch Gmünd'],
            'person_ctry_code': ['GB', 'DE'],
            'psn_name': ['FISH I AM', 'JONES FRED'],
            'psn_sector': ['COMPANY', 'INDIVIDUAL']},
            person_unused_columns)
        # duplicates added
        df_person3 = create_df_with_unused_columns({
            'person_id': ['76', '8'],
            'person_address': ['Ontario N2L 3W8', 'home'],
            'person_ctry_code': ['US', 'NZ'],
            'psn_name': ['A N OTHER', 'FISH I AM'],
            'psn_sector': ['INDIVIDUAL', 'COMPANY']},
            person_unused_columns)

        zip_file_extension = '.zip'
        file_prefix = 'tls'
        input_folder_name = os.path.join('tests', 'data')
        output_folder_name = os.path.join('tests', 'output')
        patstat_tables_file_base_name = os.path.join(input_folder_name, file_prefix)

        zip_file_names = [
            patstat_tables_file_base_name + '203_part01' + zip_file_extension,
            patstat_tables_file_base_name + '203_part02' + zip_file_extension,

            patstat_tables_file_base_name + '201_part01' + zip_file_extension,
            patstat_tables_file_base_name + '201_part02' + zip_file_extension,

            patstat_tables_file_base_name + '224_part01' + zip_file_extension,
            patstat_tables_file_base_name + '224_part02' + zip_file_extension,

            patstat_tables_file_base_name + '207_part01' + zip_file_extension,
            patstat_tables_file_base_name + '207_part02' + zip_file_extension,

            patstat_tables_file_base_name + '206_part01' + zip_file_extension,
            patstat_tables_file_base_name + '206_part02' + zip_file_extension,
            patstat_tables_file_base_name + '206_part03' + zip_file_extension,
            patstat_tables_file_base_name + '206_part04' + zip_file_extension,

            patstat_tables_file_base_name + '202_part01' + zip_file_extension,
            patstat_tables_file_base_name + '202_part02' + zip_file_extension
        ]

        text_file_extension = '.txt'
        file_names = {
            file_prefix + '203_part01' + text_file_extension: df_abstract0,
            file_prefix + '203_part02' + text_file_extension: df_abstract1,

            file_prefix + '201_part01' + text_file_extension: df_applications0,
            file_prefix + '201_part02' + text_file_extension: df_applications1,

            file_prefix + '224_part01' + text_file_extension: df_cpc0,
            file_prefix + '224_part02' + text_file_extension: df_cpc1,

            file_prefix + '207_part01' + text_file_extension: df_personapp0,
            file_prefix + '207_part02' + text_file_extension: df_personapp1,

            file_prefix + '206_part01' + text_file_extension: df_person0,
            file_prefix + '206_part02' + text_file_extension: df_person1,
            file_prefix + '206_part03' + text_file_extension: df_person2,
            file_prefix + '206_part04' + text_file_extension: df_person3,

            file_prefix + '202_part01' + text_file_extension: df_title0,
            file_prefix + '202_part02' + text_file_extension: df_title1,
        }

        pickled_dfs = {}

        def to_pickle(df, pickle_file_name):
            pickled_dfs[pickle_file_name] = df.copy(deep=True)

        def read_pickle(pickle_file_name):
            return pickled_dfs[pickle_file_name]

        def is_file(file_name):
            return file_name in zip_file_names

        class StubbedFile(object):
            def __init__(self, file_name):
                self.__file_name = file_name
                self.__df = file_names[file_name]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def df(self):
                return self.__df

        def read_csv(stubbed_file):
            return stubbed_file.df()

        class StubbedZipFile(object):
            def __init__(self, file_name):
                if file_name not in zip_file_names:
                    raise ValueError(f'{file_name} not found')

                self.file_name = file_name
                self.zipped_text_file_name = os.path.splitext(os.path.basename(self.file_name))[0] + '.txt'

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def namelist(self):
                return [self.zipped_text_file_name]

            def open(self, file_name_in_archive):
                if file_name_in_archive != self.zipped_text_file_name:
                    raise ValueError(f'{file_name_in_archive} not in zip - expected {self.zipped_text_file_name}')
                return StubbedFile(file_name_in_archive)

        return input_folder_name, output_folder_name, is_file, StubbedZipFile, read_csv, read_pickle, to_pickle

    def extractFullPatstat(self):
        input_folder_name, output_folder_name, is_file, StubbedZipFile, read_csv, read_pickle, to_pickle = self.setup()
        return convert_patstat_data_to_data_frame(input_folder_name, output_folder_name, False, None, 1.0, None,
                                                  is_file, StubbedZipFile,
                                                  read_csv, read_pickle, to_pickle)

    def extractLitePatstat(self, date_range=None):
        input_folder_name, output_folder_name, is_file, StubbedZipFile, read_csv, read_pickle, to_pickle = self.setup()
        return convert_patstat_data_to_data_frame(input_folder_name, output_folder_name, True, None, 1.0, date_range,
                                                  is_file, StubbedZipFile,
                                                  read_csv, read_pickle, to_pickle)

    def test_reads_patent_grouped_by_family(self):
        df = self.extractFullPatstat()
        self.assertEqual(3, df.shape[0])

    def test_reads_patent_abstract_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertEqual("abstract for patent appln_id 0", df.loc['family0'].abstract)
        self.assertEqual("abstract for patent appln_id 02 longest blah", df.loc['family45'].abstract)
        self.assertEqual("abstract for patent appln_id 09", df.loc['family123'].abstract)

    def test_reads_one_per_family_country_priority(self):
        df = self.extractFullPatstat()
        self.assertEqual(['family0', 'family45', 'family123'], df.patent_id.tolist())

    def test_reads_application_id_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertEqual('app_orig0', df.loc['family0'].application_id)
        self.assertEqual('app_orig02', df.loc['family45'].application_id)
        self.assertEqual('apporig_09', df.loc['family123'].application_id)

    def test_reads_application_date_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertEqual(pd.Timestamp('1999-10-17 00:00:00'), df.loc['family0'].application_date)
        self.assertEqual(pd.Timestamp('2007-02-28 00:00:00'), df.loc['family45'].application_date)
        self.assertEqual(pd.Timestamp('2010-10-01 00:00:00'), df.loc['family123'].application_date)

    def test_reads_publication_date_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertEqual(pd.Timestamp('1999-11-21 00:00:00'), df.loc['family0'].publication_date)
        self.assertEqual(pd.Timestamp('2007-07-08 00:00:00'), df.loc['family45'].publication_date)
        self.assertEqual(pd.Timestamp('2011-02-12 00:00:00'), df.loc['family123'].publication_date)

    def test_reads_application_title_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].invention_title)
        self.assertEqual('Title 2', df.loc['family45'].invention_title)
        self.assertEqual('Title of patent #09', df.loc['family123'].invention_title)

    def test_reads_cpc_codes_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].classifications_cpc)
        self.assertEqual(['H01L  24/03', 'Y02L2224/85203'], df.loc['family45'].classifications_cpc)
        self.assertEqual(['Q99Q 123/456'], df.loc['family123'].classifications_cpc)

    def test_reads_inventor_names_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].inventor_names)
        self.assertEqual(['A N OTHER', 'JONES FRED'], df.loc['family45'].inventor_names)
        self.assertEqual(['JONES FRED'], df.loc['family123'].inventor_names)

    def test_reads_inventor_countries_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].inventor_countries)
        self.assertEqual(['US', 'DE'], df.loc['family45'].inventor_countries)
        self.assertEqual(['DE'], df.loc['family123'].inventor_countries)

    def test_reads_inventor_cities_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].inventor_cities)
        self.assertEqual(['Ontario N2L 3W8', '73527 Schwäbisch Gmünd'], df.loc['family45'].inventor_cities)
        self.assertEqual(['73527 Schwäbisch Gmünd'], df.loc['family123'].inventor_cities)

    def test_reads_applicant_names_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].applicant_organisation)
        self.assertEqual(['FISH I AM'], df.loc['family45'].applicant_organisation)
        self.assertEqual(['SMITH INDUSTRIES'], df.loc['family123'].applicant_organisation)

    def test_reads_applicant_countries_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].applicant_countries)
        self.assertEqual(['GB'], df.loc['family45'].applicant_countries)
        self.assertEqual(['AU'], df.loc['family123'].applicant_countries)

    def test_reads_applicant_cities_preferred_country_in_family(self):
        df = self.extractFullPatstat()
        self.assertIsNaN(df.loc['family0'].applicant_cities)
        self.assertEqual(['Newport NP20 1XJ'], df.loc['family45'].applicant_cities)
        self.assertEqual(['somewhere else'], df.loc['family123'].applicant_cities)

    # def test_reads_first_citations(self):
    #     df = self.extractFullPatstat()_first_row
    #     self.assertEqual(2, df.appln_id.item())
    #     self.assertEqual(['C11B   3/12', 'C11B  11/005', 'C11B  13/00', 'Y02W  30/74'],
    #                      df.classifications_cpc)

    def test_lite_reads_patent_abstract_without_filtering(self):
        df = self.extractLitePatstat()
        self.assertEqual("abstract for patent appln_id 0", df.loc['family0'].abstract)
        self.assertEqual('abstract for patent appln_id 01 longest here', df.loc['family45'].abstract)
        self.assertEqual('abstract for patent appln_id 09', df.loc['family123'].abstract)

    def test_lite_reads_patent_publication_date(self):
        df = self.extractLitePatstat()
        self.assertEqual(pd.Timestamp('1999-11-21 00:00:00'), df.loc['family0'].publication_date)
        self.assertEqual(pd.Timestamp('1999-11-23 00:00:00'), df.loc['family45'].publication_date)
        self.assertEqual(pd.Timestamp('2011-02-12 00:00:00'), df.loc['family123'].publication_date)
        self.assertEqual(3, df.shape[0])

    def test_lite_reads_patent_abstract_with_from_date_filtering(self):
        df = self.extractLitePatstat(date_range=[pd.to_datetime('1999-11-22'), pd.to_datetime('today')])
        self.assertEqual('abstract for patent appln_id 01 longest here', df.loc['family45'].abstract)
        self.assertEqual('abstract for patent appln_id 09', df.loc['family123'].abstract)
        self.assertEqual(2, df.shape[0])

    def test_lite_reads_patent_abstract_with_from_date_filtering_inclusive_range(self):
        df = self.extractLitePatstat(date_range=[pd.to_datetime('1999-11-21'), pd.to_datetime('today')])
        self.assertEqual("abstract for patent appln_id 0", df.loc['family0'].abstract)
        self.assertEqual('abstract for patent appln_id 01 longest here', df.loc['family45'].abstract)
        self.assertEqual('abstract for patent appln_id 09', df.loc['family123'].abstract)
        self.assertEqual(3, df.shape[0])

    def test_lite_reads_patent_abstract_with_to_date_filtering(self):
        df = self.extractLitePatstat(date_range=[pd.to_datetime('1900-01-01'), pd.to_datetime('2011-02-11')])
        self.assertEqual("abstract for patent appln_id 0", df.loc['family0'].abstract)
        self.assertEqual('abstract for patent appln_id 01 longest here', df.loc['family45'].abstract)
        self.assertEqual(2, df.shape[0])

    def test_lite_reads_patent_abstract_with_from_to_filtering_inclusive_range(self):
        df = self.extractLitePatstat(date_range=[pd.to_datetime('1900-01-01'), pd.to_datetime('2011-02-12')])
        self.assertEqual("abstract for patent appln_id 0", df.loc['family0'].abstract)
        self.assertEqual('abstract for patent appln_id 01 longest here', df.loc['family45'].abstract)
        self.assertEqual('abstract for patent appln_id 09', df.loc['family123'].abstract)
        self.assertEqual(3, df.shape[0])

    def test_lite_reads_patent_abstract_with_from_and_to_date_filtering(self):
        df = self.extractLitePatstat(date_range=[pd.to_datetime('1999-11-22'), pd.to_datetime('2011-02-11')])
        self.assertEqual('abstract for patent appln_id 01 longest here', df.loc['family45'].abstract)
        self.assertEqual(1, df.shape[0])

    def test_lite_stores_specific_columns(self):
        df = self.extractLitePatstat()
        self.assertListEqual(['publication_date', 'patent_id', 'abstract', 'classifications_cpc'], df.columns.tolist())
