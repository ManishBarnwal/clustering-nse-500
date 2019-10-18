import luigi

import pandas as pd
import time
from datetime import date

import sys

# TODO: figure out a better way to do this
# adding this to path so that local importing of modules work
sys.path.append('/Users/manishb-imac/personal-projects/clustering-nse-500/modular_code')

from data_input.scrape_screener import ScrapeScreener

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class ScrapeCompaniesInfo(luigi.Task):

    def requires(self):
        return []

    def output(self):
        output_file_name = 'companies_info' + '_' + str(date.today()) + '.csv'
        return luigi.LocalTarget('../output_files/{}'.format(output_file_name))

    def run(self):
        time.sleep(10)
        LOG.info('--- Read NSE500 companies names ---')
        nse_500 = pd.read_csv('../input_files/ind_nifty500list.csv')
        num_companies = nse_500.Symbol.nunique()
        companies = nse_500['Symbol'].values[:num_companies]

        LOG.info('--- Loaded successfully. No. of unique companies in data: {} ---'.format(num_companies))

        # for testing code; remove later
        # num_companies_tmp = 100
        # companies = nse_500['Symbol'].values[:num_companies_tmp]

        LOG.info('--- Starting to scrape data ---')

        companies_info_scraped = ScrapeScreener.scrape_data(companies, type_data='stand_alone')

        cols_to_clean = ['market_cap', 'current_price', 'book_value', 'stock_pe',
                         'dividend_yield', 'roce', 'roe', 'sales_growth_3yrs', 'face_value']

        companies_info_final = ScrapeScreener.clean_data(companies_info_scraped, cols_to_clean)
        LOG.info('--- Successfully scraped companies information ---')

        with self.output().open('w') as out_file:
            companies_info_final.to_csv(out_file, index=False)


class HandleMissingData(luigi.Task):
    def requires(self):
        return ScrapeCompaniesInfo()

    def output(self):
        output_file_name = 'all_companies_info' + '_' + str(date.today()) + '.csv'
        return luigi.LocalTarget('../output_files/{}'.format(output_file_name))

    def run(self):
        with self.input().open('r') as infile:
            companies_info = pd.read_csv(infile)
            print('companies_info.shape: ', companies_info.shape)

        # companies having missing stock_pe along with negative sales_growth are bad companies
        # and their missing stock_pe can be filled by -1
        companies_info = self.impute_bad_pe(companies_info,
                                            second_col='sales_growth_3yrs',
                                            fill_value=-1)  # impute actually-bad-performing stock_pe with -1

        # we'll try to impute stock_pe for the companies with missing stock_pe by scraping their consolidated data
        missing_stock_pe_bool = companies_info['stock_pe'].isna()
        consolidated_company_names = companies_info.loc[missing_stock_pe_bool, 'symbol'].values
        LOG.info('--- consolidated_company_names: ', consolidated_company_names)

        # drop these companies as we will get their consolidated data
        missing_stock_pe_row_ind = missing_stock_pe_bool.index[missing_stock_pe_bool]
        companies_info_standalone = companies_info.drop(missing_stock_pe_row_ind)

        LOG.info('--- Scraping consolidated data ---')
        consolidated_company_info = ScrapeScreener.scrape_data(companies_list=consolidated_company_names,
                                                               type_data='consolidated')

        # TODO: parametrize this in luigi
        cols_to_clean = ['market_cap', 'current_price', 'book_value', 'stock_pe',
                         'dividend_yield', 'roce', 'roe', 'sales_growth_3yrs', 'face_value']

        consolidated_company_info_cleaned = ScrapeScreener.clean_data(consolidated_company_info, cols_to_clean)

        all_companies_info = pd.concat([companies_info_standalone, consolidated_company_info_cleaned], axis=0)

        # companies having missing stock_pe along with negative roe are bad companies
        # and their missing stock_pe can be filled by -1
        all_companies_info = self.impute_bad_pe(all_companies_info, second_col='roe',
                                                fill_value=-1)  # impute actually-bad-performing stock_pe with -1
        LOG.info('--- Successfully scraped all companies information ---')
        LOG.info('---No. of companies in total: {} ---'.format(all_companies_info.shape))

        with self.output().open('w') as outfile:
            all_companies_info.to_csv(outfile,  index=False)

    @staticmethod
    def impute_bad_pe(df, second_col, fill_value=-1):
        missing_stock_pe_ind = df.stock_pe.isna()
        neg_second_col = df[second_col] < 0
        bad_pe_ind = missing_stock_pe_ind & neg_second_col

        df.loc[bad_pe_ind, 'stock_pe'] = fill_value

        return df


class ImputeMissingData(luigi.Task):
    def requires(self):
        return HandleMissingData()

    def output(self):
        output_file_name = 'companies_info_imputed' + '_' + str(date.today()) + '.csv'
        return luigi.LocalTarget('../output_files/{}'.format(output_file_name))

    def run(self):
        with self.input().open('r') as infile:
            companies_info = pd.read_csv(infile)

        LOG.info('--- Loaded successfully. No. of companies in data ---: {}'.format(companies_info.shape))

        companies_info_imputed = self.impute_missing_values(companies_info)

        LOG.info('--- No. of companies after dropping missing values ---: {}'.format(companies_info_imputed.shape))

        with self.output().open('w') as outfile:
            companies_info_imputed.to_csv(outfile)

    @staticmethod
    def impute_missing_values(df):
        missing_roe = df.roe.isna()
        # replace missing roe with 0
        df.loc[missing_roe, 'roe'] = 0

        # replace missing roe with 0
        missing_sales_growth = df.sales_growth_3yrs.isna()
        df.loc[missing_sales_growth, 'sales_growth_3yrs'] = 0

        df = df.dropna(how='any')

        return df


if __name__ == '__main__':
    luigi.run()
