from collections import Counter

import numpy as np
import pandas as pd

import spacy
import luigi

import os
import sys
import logging
from datetime import date


# TODO: figure out a better way to do this
# adding this to path so that local importing of modules work
sys.path.append('/Users/manishb-imac/personal-projects/clustering-nse-500/modular_code')

from data_input.scrape_about_company import ScrapeAboutCompanyText

logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)

LOG = logging.getLogger(__name__)


class Word2VecEmbeddings(luigi.Task):
    cols_to_clean_default = ['market_cap', 'current_price', 'book_value', 'stock_pe',
                             'dividend_yield', 'roce', 'roe', 'sales_growth_3yrs', 'face_value']

    cols_to_clean = luigi.ListParameter(default=cols_to_clean_default)
    output_dir = luigi.Parameter(default='../output_files/')
    output_filename = luigi.Parameter(default='text_embeddings.csv')

    language = luigi.Parameter(default='en_core_web_md')

    def requires(self):
        return ScrapeAboutCompanyText(
            output_dir=self.output_dir,
            cols_to_clean=self.cols_to_clean
        )

    def output(self):
        return luigi.LocalTarget(self.output_path)

    @property
    def output_path(self):
        return os.path.join(
            self.output_dir,
            '{}'.format(date.today()),
            '{}'.format(self.output_filename)
        )

    def run(self):

        with self.input().open('r') as infile:
            about_company_df = pd.read_csv(infile)
            LOG.info(f'--- Successfully loaded about-text company data.')

        language = spacy.load(self.language)
        LOG.info(f'--- Generating term embeddings for {about_company_df.shape[0]} companies ---')

        doc_vectors = (
            about_company_df.pipe(self._get_tfidf_weights, language=language)
                .pipe(self._doc_vectors, language=language)
                .pipe(self._normalize)
                .dropna(axis='index')
        )

        LOG.info(f'--- Retaining {len(doc_vectors)} companies with text embeddings ---')

        # assigning meaningful names for the embeddings columns
        embedding_columns = ['embedding_' + str(col) for col in doc_vectors.columns if col != 'symbol']
        doc_vectors.columns = embedding_columns

        with self.output().open('w') as outfile:
            doc_vectors.reset_index().to_csv(outfile, index=False)

    def _get_tfidf_weights(self, docs, language):

        docs_dict = docs.set_index('symbol')['about_company'].to_dict()  # _parse_doc_tokens expects a dictionary
        term_counts = pd.DataFrame.from_records(
            data=self._parse_doc_tokens(docs_dict, language),
            columns=['symbol', 'term', 'frequency']
        )

        if term_counts.empty:
            raise ValueError('No documents parsed successfully. Did you load the spacy language file successfully?')

        LOG.debug('Calculating term frequency')
        term_frequency = term_counts.set_index(['symbol', 'term'])['frequency']
        term_frequency = (1 + term_frequency / term_frequency.groupby(level='symbol').max()) / 2

        LOG.debug('Calculating document frequency')
        num_docs = term_counts['symbol'].nunique()
        idoc_frequency = np.log(num_docs / term_counts.groupby('term')['symbol'].nunique())

        LOG.debug('Calculating tf-idf')
        return term_frequency.mul(idoc_frequency, level='term').rename('weight')

    @staticmethod
    def _parse_doc_tokens(docs, language):
        for symbol, doc in docs.items():
            parsed = language(doc.lower())
            lemmas = [x.lemma_ for x in parsed if x.has_vector]
            counts = Counter(lemmas)
            for term, frequency in counts.items():
                yield symbol, term, frequency

    @staticmethod
    def _doc_vectors(tfidf, language):
        terms = tfidf.index.get_level_values('term').unique()

        LOG.debug('Calculating term vectors')
        term_vectors = pd.DataFrame.from_dict(
            data={term: language(term).vector for term in terms},
            orient='index'
        )

        LOG.debug('Calculating company vectors')
        return term_vectors.mul(tfidf, axis=0, level='term').groupby(level='symbol').mean()

    @staticmethod
    def _normalize(df):
        norms = (df ** 2).sum('columns') ** 0.5
        return df.div(norms, axis='index')
