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
from sklearn.preprocessing import MinMaxScaler
from models.kmeans_clustering import KMeansClustering


logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class NSECompaniesSegmentation(luigi.Task):

    cols_to_clean_default = ['market_cap', 'current_price', 'book_value', 'stock_pe', 'dividend_yield',
                             'roce', 'roe', 'sales_growth_3yrs', 'face_value']

    cols_to_clean = luigi.ListParameter(default=cols_to_clean_default)
    output_dir = luigi.Parameter(default='../output_files/')
    n_clusters = luigi.IntParameter(default=3)

    @property
    def output_path(self):
        return os.path.join(
            self.output_dir,
            'companies_segmentation_',
            '{}'.format(str(date.today())),
            '.csv'
        )

    def requires(self):
        return ImputeMissingData(cols_to_clean=self.cols_to_clean,
                                 output_dir=self.output_dir)

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        with self.input().open('r') as infile:
            companies_info = pd.read_csv(infile)
            LOG.info('---Loaded imputed companies data successfully ---')

        LOG.info('--- No. of companies in data: {} ---'.format(companies_info.shape[0]))

        df_normalized = self.scale_df(companies_info)

        kmeans_model = KMeansClustering(n_clusters=self.n_clusters)
        kmeans_model.fit(df_normalized)
        kmeans_cluster = kmeans_model.predict(df_normalized)

        company_cluster = companies_info.copy()
        company_cluster['cluster'] = kmeans_cluster

        with self.output().open('w') as outfile:
            company_cluster.to_csv(outfile, index=False)

    @staticmethod
    def scale_df(df):
        df_numeric = df.select_dtypes(include=['float'])
        scaler = MinMaxScaler()

        df_normalized = scaler.fit_transform(df_numeric)

        df_normalized = pd.DataFrame(df_normalized, columns=df_numeric.columns)

        return df_normalized


if __name__ == '__main__':
    luigi.run()


