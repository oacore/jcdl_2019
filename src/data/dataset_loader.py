"""
Utility functions for processing the CORE-CrossRef dataset
"""

import csv
import json
import logging
from datetime import datetime
from typing import MutableMapping, Any

import pandas as pd
from tqdm import tqdm
from pandas import DataFrame

__author__ = 'Dasha Herrmannova'
__email__ = 'dasha.herrmannova@open.ac.uk'


Row = MutableMapping[str, Any]


class DatasetLoader(object):

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def _has_issn(self, data_row: Row) -> bool:
        """Check if 'data_row' contains at least one ISSN number
        
        :param data_row: one data row from the input CSV
        :type data_row: Row
        :return: true if 'data_row' contains at least one ISSN number, false 
                 otherwise
        :rtype: bool
        """
        for issn in data_row['cr_issn']:    
            if issn and len(issn) and issn != '' and issn != '\\N':
                return True
        return False

    def load_dataset(self, csv_path: str) -> DataFrame:
        """Load final dataset
        
        :param csv_path: [description]
        :type csv_path: str
        :return: [description]
        :rtype: DataFrame
        """
        data = []
        self._logger.info('Loading dataset from {}'.format(csv_path))
        
        date_cols = [
            'core_metadata_added', 'core_deposited_date', 'cr_published', 
            'cr_created', 'cr_accepted'
        ]

        with open(csv_path) as fp:
            reader = csv.DictReader(fp, delimiter=',')
            for row in tqdm(reader):
                data_row: Row = {
                    k: json.loads(v) for k, v in row.items()
                }
                for k in date_cols:
                    dates = data_row[k]
                    if isinstance(dates, list):
                        dates = [
                            datetime.strptime(x, '%Y-%m-%d') if x else x 
                            for x in dates
                        ]
                    else:
                        dates = (
                            datetime.strptime(dates, '%Y-%m-%d') 
                            if dates else dates
                        )
                    data_row[k] = dates
                data_row['is_gb'] = 'gb' in data_row['core_country_code']
                data_row['has_issn'] = self._has_issn(data_row)
                data_row['core_country_code'] = [
                    'uk' if x == 'gb' else x 
                    for x in data_row['core_country_code']
                ]
                data.append(data_row)
        df = DataFrame(data)
        self._logger.info('Done loading, got {} documents'.format(len(df)))
        self._logger.debug('Dataset schema: {}'.format(df.dtypes))
        return df
