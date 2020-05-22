import requests as rq
from bs4 import BeautifulSoup as bs

import numpy as np
import pandas as pd

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)  # setting level to debug; will log all types of logging
LOG = logging.getLogger(__name__)


class ScrapeScreener:

    @staticmethod
    def scrape_data(companies_list, type_data='stand_alone'):
        final_basic_stats_list = []
        for company in companies_list:
            if type_data == 'stand_alone':
                url = f'https://www.screener.in/company/{company}'
            elif type_data == 'consolidated':
                url = f'https://www.screener.in/company/{company}/{type_data}'
            try:

                response = rq.get(url)
                soup = bs(response.text, "html.parser")  # parse the html page
                basic_features_soup = soup.find_all(class_='row-full-width')
                basic_features_list = basic_features_soup[0].find_all(class_='four columns')
                basic_stats = [f.get_text() for f in basic_features_list]

                basic_stats = [f.lower().strip().replace('\n', '').replace('  ', '').replace(' ', '_') for f in basic_stats]
                company_stats_dict = dict()
                company_stats_dict['symbol'] = company
                for f in basic_stats:
                    s = f.split(":")
                    if len(s) == 2:
                        company_stats_dict[s[0]] = s[1]
                final_basic_stats_list.append(list(company_stats_dict.values()))
            except IndexError:
                LOG.exception(f'--- Error in scraping {company} company data. Continue to scrape. ---')
                pass

        company_stats_df = pd.DataFrame(final_basic_stats_list,
                                        columns=company_stats_dict.keys())
        change_col_names = {'stock_p/e': 'stock_pe',
                            'sales_growth_(3yrs)': 'sales_growth_3yrs'
                            }
        company_stats_df.rename(change_col_names, axis=1, inplace=True)

        return company_stats_df

    @staticmethod
    def clean_data(df, cols_to_process):
        for col in cols_to_process:
            df[col] = df[col].apply(lambda x: x.replace('cr.', '').replace(',', '').replace('%', ''))

        # catch missing values
        df = df.replace('cr.', np.NaN).replace('%', np.NaN).replace('', np.NaN)

        # change column data-type to float
        cols_type = {}
        for col in cols_to_process:
            cols_type[col] = 'float32'

        df = df.astype(cols_type)  # change object type column to float

        for col in cols_to_process:
            df[col] = df[col].apply(lambda x: np.round(x, 2))

        return df

    @staticmethod
    def scrape_about_company_info(companies_list, type_data='stand_alone'):
        final_about_company_list = []
        for company in companies_list:
            if type_data == 'stand_alone':
                url = f'https://www.screener.in/company/{company}'
            elif type_data == 'consolidated':
                url = f'https://www.screener.in/company/{company}/{type_data}'
            try:

                response = rq.get(url)

                soup = bs(response.text, "lxml")  # parse the page

                about_company_html = soup.find_all(id='company-profile')
                about_company_text = str(about_company_html[0])
                about_company_text = about_company_text.split(sep='">')[1]
                about_company_final = about_company_text.split('(Source')[0]

                about_company_dict = dict()
                about_company_dict['symbol'] = company
                about_company_dict['about_company'] = about_company_final

                final_about_company_list.append(list(about_company_dict.values()))
            except IndexError:
                LOG.exception(f'Error in scraping {company} company data. Continue to scrape.')
                pass

        about_company_df = pd.DataFrame(final_about_company_list,
                                        columns=about_company_dict.keys())

        return about_company_df
