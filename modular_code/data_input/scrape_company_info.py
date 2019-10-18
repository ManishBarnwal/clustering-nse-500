import requests as rq
import luigi
from bs4 import BeautifulSoup as bs

import numpy as np
import pandas as pd
import time
from datetime import date

# TODO: figure out how to inlcude the class here
# from data_input.scrape_screener import ScrapeScreener

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class ScrapeCompaniesInfo(luigi.Task):

    def requires(self):
        return []

    def output(self):
        output_file_name = 'companies_info_data' + '_' + str(date.today()) + '.csv'
        return luigi.LocalTarget('../output_files/{}'.format(output_file_name))

    def run(self):
        time.sleep(10)
        LOG.info('--- Read NSE500 companies names ---')
        nse_500 = pd.read_csv('../input_files/ind_nifty500list.csv')
        num_companies = nse_500.Symbol.nunique()
        LOG.info('--- Loaded successfully. No. of unique companies in data: {} ---'.format(num_companies))

        # num_companies_tmp = 100
        companies = nse_500['Symbol'].values[:num_companies]

        LOG.info('--- Starting to scrape data ---')

        companies_info_scraped = ScrapeScreener.scrape_data(companies, type_data='stand_alone')

        cols_to_clean = ['market_cap', 'current_price', 'book_value', 'stock_pe',
                         'dividend_yield', 'roce', 'roe', 'sales_growth_3yrs', 'face_value']

        for col in cols_to_clean:
            companies_info_scraped[col] = companies_info_scraped[col].apply(lambda x: ScrapeScreener.clean_values(x))

        companies_info = ScrapeScreener.catch_missing_values(companies_info_scraped)
        companies_info_final = ScrapeScreener.change_to_float_type(companies_info, cols_to_clean)

        LOG.info('--- Successfully scraped companies information ---')

        with self.output().open('w') as out_file:
            companies_info_final.to_csv(out_file, index=False)

    class HandleMissingData(luigi.Task):
        def requires(self):
            return ScrapeCompaniesInfo()

        def output(self):
            output_file_name = 'companies_info_all' + '_' + str(date.today()) + '.csv'
            return luigi.LocalTarget('../output_files/{}'.format(output_file_name))

        def run(self):
            with self.input().open('r') as infile:
                companies_info = pd.read_csv(infile)
                print('companies_info.shape: ', companies_info.shape)

            # companies having missing stock_pe along with negative sales_growth are bad companies
            # and their missing stock_pe can be filled by -1
            companies_info = self.impute_bad_pe(companies_info, second_col='sales_growth_3yrs',
                                                fill_value=-1)  # impute actually-bad-performing stock_pe with -1

            # we'll try to impute stock_pe for the companies with missing stock_pe by scraping their consolidated data
            missing_stock_pe_bool = companies_info['stock_pe'].isna()
            consolidated_company_names = companies_info.loc[missing_stock_pe_bool, 'symbol'].values
            print('consolidated_company_names: ', consolidated_company_names)

            # drop these companies as we will get their consolidated data
            missing_stock_pe_row_ind = missing_stock_pe_bool.index[missing_stock_pe_bool]
            companies_info_standalone = companies_info.drop(missing_stock_pe_row_ind)

            LOG.info('--- Shape of companies_info after dropping consolidated companies --- {}'.format(
                companies_info.shape))

            LOG.info('--- Scraping consolidated data ---')
            consolidated_company_info = ScrapeScreener.scrape_data(companies_list=consolidated_company_names,
                                                                   type_data='consolidated')

            cols_to_clean = ['market_cap', 'current_price', 'book_value', 'stock_pe',
                             'dividend_yield', 'roce', 'roe', 'sales_growth_3yrs', 'face_value']

            for col in cols_to_clean:
                consolidated_company_info[col] = consolidated_company_info[col].apply(
                    lambda x: ScrapeScreener.clean_values(x))

            consolidated_company_info = ScrapeScreener.catch_missing_values(consolidated_company_info)
            consolidated_company_info_final = ScrapeScreener.change_to_float_type(consolidated_company_info, cols_to_clean)

            all_companies_info = pd.concat([companies_info_standalone, consolidated_company_info_final], axis=0)

            # companies having missing stock_pe along with negative roe are bad companies
            # and their missing stock_pe can be filled by -1
            all_companies_info = self.impute_bad_pe(all_companies_info, second_col='roe',
                                                    fill_value=-1)  # impute actually-bad-performing stock_pe with -1

            print('all_companies_info.shape: {}'.format(all_companies_info.shape))

            LOG.info('--- Successfully scraped all companies information ---')

            with self.output().open('w') as outfile:
                all_companies_info.to_csv(outfile,  index=False)

        @staticmethod
        def impute_bad_pe(df, second_col, fill_value=-1):
            missing_stock_pe_ind = df.stock_pe.isna()
            neg_second_col = df[second_col] < 0
            bad_pe_ind = missing_stock_pe_ind & neg_second_col

            df.loc[bad_pe_ind, 'stock_pe'] = fill_value

            return df


class ScrapeScreener:

    @staticmethod
    def scrape_data(companies_list, type_data='stand_alone'):
        final_basic_stats_list = []
        for company in companies_list:
            if type_data == 'stand_alone':
                url = f'https://www.screener.in/company/{company}'
            elif type_data == 'consolidated':
                url = f'https://www.screener.in/company/{company}/{type_data}'
            response = rq.get(url)
            soup = bs(response.text, "html.parser")  # parse the html page
            basic_features_soup = soup.find_all(class_='row-full-width')
            basic_features_list = basic_features_soup[0].find_all(class_='four columns')
            basic_stats = [f.get_text() for f in basic_features_list]

            basic_stats = [f.lower().strip().replace('\n', '').replace('  ', '').replace(' ', '_') for f in basic_stats]

            company_stats_dict = {}
            company_stats_dict['symbol'] = company
            for f in basic_stats:
                s = f.split(":")
                if len(s) == 2:
                    company_stats_dict[s[0]] = s[1]
            final_basic_stats_list.append(list(company_stats_dict.values()))

        company_stats_df = pd.DataFrame(final_basic_stats_list,
                                        columns=company_stats_dict.keys())
        change_col_names = {'stock_p/e': 'stock_pe',
                            'sales_growth_(3yrs)': 'sales_growth_3yrs'
                            }
        company_stats_df.rename(change_col_names, axis=1, inplace=True)

        return company_stats_df

    @staticmethod
    def clean_values(x):
        return x.replace('cr.', '').replace(',', '').replace('%', '')

    @staticmethod
    def catch_missing_values(df):
        df = df.replace('cr.', np.NaN).replace('%', np.NaN).replace('', np.NaN)
        return df

    @staticmethod
    def change_to_float_type(df, cols_to_process):
        cols_type = {}
        for col in cols_to_process:
            cols_type[col] = 'float32'

        df = df.astype(cols_type)  # change object type column to float

        for col in cols_to_process:
            df[col] = df[col].apply(lambda x: np.round(x, 2))

        return df


if __name__ == '__main__':
    luigi.run()
