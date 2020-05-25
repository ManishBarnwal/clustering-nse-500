import luigi

import pandas as pd
import os
import sys
import logging
from datetime import date

from sklearn.preprocessing import MinMaxScaler

# TODO: figure out a better way to do this - somehow this is not working w/o explicitly adding the path
# adding this to path so that local importing of modules work
sys.path.append('/Users/manishb-imac/personal-projects/clustering-nse-500/modular_code')

from preprocess.merge_company_stats_embeddings import MergeCompanyStatsEmbeddings
from models.kmeans_clustering import KMeansClustering


logging.basicConfig(format='%(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)


class NSECompaniesSegmentation(luigi.Task):

    cols_to_clean_default = ['market_cap', 'current_price', 'book_value', 'stock_pe', 'dividend_yield',
                             'roce', 'roe', 'sales_growth_3yrs', 'face_value']

    cols_to_clean = luigi.ListParameter(default=cols_to_clean_default)
    output_dir = luigi.Parameter(default='../output_files/')
    output_filename = luigi.Parameter(default='companies_segments.csv')
    n_clusters = luigi.IntParameter(default=3)

    @property
    def output_path(self):
        return os.path.join(
            self.output_dir,
            '{}'.format(date.today()),
            '{}'.format(self.output_filename)
        )

    def requires(self):
        return MergeCompanyStatsEmbeddings(
            cols_to_clean=self.cols_to_clean,
            output_dir=self.output_dir
        )

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        with self.input().open('r') as infile:
            company_stats_embeddings = pd.read_csv(infile)
            LOG.info('--- Successfully loaded merged company stats and embeddings data ---')
            LOG.info(f'--- No. of companies in data: {company_stats_embeddings.shape[0]} ---')

        df_normalized = self.scale_df(company_stats_embeddings)

        LOG.info('--- Starting k-means clustering ---')
        kmeans_model = KMeansClustering(n_clusters=self.n_clusters)
        kmeans_model.fit(df_normalized)
        kmeans_cluster = kmeans_model.predict(df_normalized)

        company_cluster = company_stats_embeddings.copy()
        company_cluster['cluster'] = kmeans_cluster

        LOG.info('--- Distribution of clusters ---')
        LOG.info(f'--- \n{company_cluster.cluster.value_counts()} \n ---')

        LOG.info('--- Dumping generated clusters ---')
        with self.output().open('w') as outfile:
            company_cluster.to_csv(outfile, index=False)

    @staticmethod
    def scale_df(df):
        df_numeric = df.select_dtypes(include=['float', 'int'])
        scaler = MinMaxScaler()

        df_normalized = scaler.fit_transform(df_numeric)
        df_normalized = pd.DataFrame(df_normalized, columns=df_numeric.columns)
        return df_normalized


if __name__ == '__main__':
    luigi.run()


