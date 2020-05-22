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
from preprocess.text_embeddings import Word2VecEmbeddings


logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class MergeCompanyStatsEmbeddings(luigi.Task):
    cols_to_clean_default = ['market_cap', 'current_price', 'book_value', 'stock_pe',
                             'dividend_yield', 'roce', 'roe', 'sales_growth_3yrs', 'face_value']
    cols_to_clean = luigi.ListParameter(default=cols_to_clean_default)
    output_dir = luigi.Parameter(default='../output_files/')
    output_filename = luigi.Parameter(default='companies_stats_embeddings.csv')

    @property
    def output_path(self):
        return os.path.join(
            self.output_dir,
            '{}'.format(date.today()),
            '{}'.format(self.output_filename)
        )

    def requires(self):

        return {
            'company_stats': ImputeMissingData(
                output_dir=self.output_dir,
                cols_to_clean=self.cols_to_clean
            ),

            'company_embeddings': Word2VecEmbeddings(
                output_dir=self.output_dir,
                cols_to_clean=self.cols_to_clean
            )
        }

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        with self.input()['company_stats'].open('r') as infile:
            companies_stats = pd.read_csv(infile)
            LOG.info(f'--- Successfully loaded companies info. data ---')
            LOG.info(f'--- No. of companies in data : {companies_stats.shape} ---')

        with self.input()['company_embeddings'].open('r') as infile:
            company_embeddings = pd.read_csv(infile)
            LOG.info(f'--- Successfully loaded embeddings data ---')
            LOG.info(f'--- No. of companies in data: {company_embeddings.shape[0]}')

        LOG.info(f'--- Merging company stats and embeddings data ---')
        company_stats_embeddings = pd.merge(left=companies_stats,
                                            right=company_embeddings,
                                            on=['symbol'], how='inner')

        LOG.info(f'--- Successfully merged company stats and embeddings data ---')
        LOG.info(f'No. of rows and columns in merged data: {company_stats_embeddings.shape} ---')

        LOG.info(f'--- Dumping merged data ---')
        with self.output().open('w') as outfile:
            company_stats_embeddings.to_csv(outfile, index=False)
