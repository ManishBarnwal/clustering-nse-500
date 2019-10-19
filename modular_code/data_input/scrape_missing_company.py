import luigi

import pandas as pd

import os
import logging
from datetime import date

from data_input.scrape_company_info import ScrapeCompaniesInfo
from data_input.scrape_screener import ScrapeScreener


logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class ScrapeMissingCompanies(luigi.Task):
    cols_to_clean = luigi.ListParameter()
    output_dir = luigi.Parameter(default='../output_files/')

    def requires(self):
        return ScrapeCompaniesInfo(cols_to_clean=self.cols_to_clean,
                                   output_dir=self.output_dir
                                  )

    def output(self):
        output_file_name = 'all_companies_info_' + str(date.today()) + '.csv'
        output_path = os.path.join(self.output_dir, output_file_name)
        return luigi.LocalTarget(output_path)

    def run(self):
        with self.input().open('r') as infile:
            companies_info = pd.read_csv(infile)

        # companies having missing stock_pe along with negative sales_growth are bad companies
        # and their missing stock_pe can be filled by -1
        companies_info = self.impute_bad_pe(companies_info,
                                            second_col='sales_growth_3yrs',
                                            fill_value=-1)  # impute actually-bad-performing stock_pe with -1

        # we'll try to impute stock_pe for the companies with missing stock_pe by scraping their consolidated data
        missing_stock_pe_bool = companies_info['stock_pe'].isna()
        consolidated_company_names = companies_info.loc[missing_stock_pe_bool, 'symbol'].values
        LOG.info('--- Consolidated_company_names-- {}'.format(consolidated_company_names))

        # drop these companies as we will get their consolidated data
        missing_stock_pe_row_ind = missing_stock_pe_bool.index[missing_stock_pe_bool]
        companies_info_standalone = companies_info.drop(missing_stock_pe_row_ind)

        LOG.info('--- Scraping consolidated data ---')
        consolidated_company_info = ScrapeScreener.scrape_data(companies_list=consolidated_company_names,
                                                               type_data='consolidated')
        consolidated_company_info_cleaned = ScrapeScreener.clean_data(consolidated_company_info, self.cols_to_clean)

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
