import luigi

import pandas as pd

import os
import sys
import logging
from datetime import date

# TODO: figure out a better way to do this
# adding this to path so that local importing of modules work
sys.path.append('/Users/manishb-imac/personal-projects/clustering-nse-500/modular_code')

from preprocess.impute_missing_data import ImputeMissingData
from data_input.scrape_screener import ScrapeScreener


logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class ScrapeAboutCompanyText(luigi.Task):
    cols_to_clean_default = ['market_cap', 'current_price', 'book_value', 'stock_pe',
                             'dividend_yield', 'roce', 'roe', 'sales_growth_3yrs', 'face_value']
    cols_to_clean = luigi.ListParameter(default=cols_to_clean_default)
    output_dir = luigi.Parameter(default='../output_files/')
    output_filename = luigi.Parameter(default='about_company_text.csv')

    @property
    def output_path(self):
        return os.path.join(
            self.output_dir,
            '{}'.format(date.today()),
            '{}'.format(self.output_filename)
        )

    def requires(self):
        return ImputeMissingData(
            output_dir=self.output_dir,
            cols_to_clean=self.cols_to_clean
        )

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        with self.input().open('r') as infile:
            companies_info = pd.read_csv(infile)
            LOG.info(f'--- Successfully loaded companies info. No. of companies in data ---: {companies_info.shape[0]}')

        company_symbol = companies_info['symbol'].values
        LOG.info('--- Scraping about company text ---')
        about_company_df = ScrapeScreener.scrape_about_company_info(company_symbol, type_data='stand_alone')
        LOG.info(f'--- First 5 rows of company information: \n {about_company_df.head(5)}')

        LOG.info(f'--- No. of companies for which about text was extracted ---: {about_company_df.shape[0]}')
        with self.output().open('w') as outfile:
            about_company_df.to_csv(outfile, index=False)
