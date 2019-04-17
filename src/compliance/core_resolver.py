
import json
import logging
from datetime import datetime
import urllib.parse as urlparse
from typing import Dict, Any, List

import requests

__author__ = 'Dasha Herrmannova'
__email__ = 'dasha.herrmannova@open.ac.uk'


class CoreResolver(object):

    def __init__(self, api_key: str, max_query_size: int = 100) -> None:
        """Initialize 
        
        :param api_key: CORE API key, can be obtained from here: 
                        https://core.ac.uk/api-keys/register/
        :type api_key: str
        :param max_query_size: maximum size of one query, defaults to 100
        :param max_query_size: int, optional
        :return: None
        :rtype: None
        """
        self._logger = logging.getLogger(__name__)
        self._endpoint = 'https://core.ac.uk/api-v2/'
        self._api_key = api_key
        self._max_query_size = max_query_size
        # disable SSL warnings
        requests.packages.urllib3.disable_warnings()

    def _process_document(self, response_doc: Dict[str, Any]) -> Dict[str, str]:
        """Process response received from CORE
        
        :param response_doc: CORE metadata
        :type response_doc: Dict[str, Any]
        :return: dictionary with the following three fields: 'doi', 'title', 
                 'published'
        :rtype: Dict[str, str]
        """
        ts = response_doc['repositoryDocument']['depositedDate']
        return {
            'id': response_doc['id'],
            'title': response_doc['title'],
            'doi': (
                response_doc['doi'] if 'doi' in response_doc else None
            ),
            'deposited': datetime.fromtimestamp(
                ts / 1000
            ).strftime('%Y-%m-%d')
        }

    def _doi_match(self, response_doc: Dict[str, Any], doi: str) -> bool:
        """Check if metadata retrieved from CORE matches the requested DOI
        
        :param response_doc: CORE metadata
        :type response_doc: Dict[str, Any]
        :param doi: DOI to match
        :type doi: str
        :return: True if document matches DOI, False otherwise
        :rtype: bool
        """
        return (
            'doi' in response_doc 
            and response_doc['doi'] 
            and response_doc['doi'].lower() == doi.lower()
        )

    def get_by_dois(self, doi_list: List[str]) -> Dict[str, Dict[str, str]]:
        """Get CORE metadata for a list of DOIs
        
        :param doi_list: DOIs to get metadata for
        :type doi_list: List[str]
        :return: dictionary in which each item is one of the supplied DOIs
                 and each value is the matching CORE metadata
        :rtype: Dict[str, Dict[str, str]]
        """
        url = urlparse.urljoin(self._endpoint, 'articles/search')

        results = {doi: None for doi in doi_list}

        # CORE API restricts query size
        split_list = [
            doi_list[i:i + self._max_query_size] 
            for i in range(0, len(doi_list), self._max_query_size)
        ]

        for sublist in split_list:
            req_data = [{'query': 'doi:"{}"'.format(d)} for d in sublist]
            headers = {'apiKey': self._api_key}
            response = requests.post(
                url, data=json.dumps(req_data), headers=headers, verify=False
            ).text
            if len(response) > 0:
                # response should contain one result dict per each DOI request
                for idx, result in enumerate(json.loads(response)):
                    if result['status'] == 'OK' and len(result['data']):
                        # results are sorted by relevance
                        document = result['data'][0]
                        if self._doi_match(document, sublist[idx]):
                            processed_doc = self._process_document(document)
                            results[processed_doc['doi']] = processed_doc
        
        return results
