import luigi

import pandas as pd

import os
import logging
from datetime import date

from data_input.scrape_screener import ScrapeScreener


logging.basicConfig(format='%(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)


class ScrapeCompaniesInfo(luigi.Task):
    cols_to_clean = luigi.ListParameter()
    output_dir = luigi.Parameter(default='../output_files/')
    output_filename = luigi.Parameter(default='companies_info.csv')

    @property
    def output_path(self):
        return os.path.join(
            self.output_dir,
            '{}'.format(date.today()),
            '{}'.format(self.output_filename)
        )

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        LOG.info('--- Reading NSE500 company names ---')
        nse_500 = pd.read_csv('../input_files/ind_nifty500list.csv')
        num_companies = nse_500.Symbol.nunique()
        companies = nse_500['Symbol'].values[:num_companies]

        LOG.info('--- Loaded successfully. No. of unique companies in data: {} ---'.format(num_companies))

        # for testing code; remove later
        # num_companies_tmp = 50
        # companies = nse_500['Symbol'].values[:num_companies_tmp]

        LOG.info('--- Starting to scrape data ---')

        companies_info_scraped = ScrapeScreener.scrape_data(companies, type_data='stand_alone')
        companies_info_final = ScrapeScreener.clean_data(companies_info_scraped, self.cols_to_clean)
        LOG.info('--- Successfully scraped companies information ---')

        with self.output().open('w') as out_file:
            companies_info_final.to_csv(out_file, index=False)
