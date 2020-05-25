import luigi

import pandas as pd

import os
import logging
from datetime import date

from data_input.scrape_missing_company import ScrapeMissingCompanies


logging.basicConfig(format='%(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)


class ImputeMissingData(luigi.Task):
    cols_to_clean = luigi.ListParameter()
    output_dir = luigi.Parameter(default='../output_files/')
    output_filename = luigi.Parameter(default='companies_info_imputed.csv')

    @property
    def output_path(self):
        return os.path.join(
            self.output_dir,
            '{}'.format(date.today()),
            '{}'.format(self.output_filename)
        )

    def requires(self):
        return ScrapeMissingCompanies(cols_to_clean=self.cols_to_clean,
                                      output_dir=self.output_dir
                                      )

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        with self.input().open('r') as infile:
            companies_info = pd.read_csv(infile)
            LOG.info(f'--- Successfully loaded. No. of companies in data: {companies_info.shape[0]} ---')

        companies_info_imputed = self.impute_missing_values(companies_info)

        LOG.info(f'--- No. of companies after dropping missing values: {companies_info_imputed.shape[0]} ---')
        with self.output().open('w') as outfile:
            companies_info_imputed.to_csv(outfile, index=False)

    @staticmethod
    def impute_missing_values(df):
        # replace missing roe with 0
        missing_roe = df.roe.isna()
        df.loc[missing_roe, 'roe'] = 0

        # replace missing roe with 0
        missing_sales_growth = df.sales_growth_3yrs.isna()
        df.loc[missing_sales_growth, 'sales_growth_3yrs'] = 0

        df = df.dropna(how='any')
        return df
