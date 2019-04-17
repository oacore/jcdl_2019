
import json
import logging
from datetime import datetime
import urllib.parse as urlparse
from typing import Dict, Any, Optional, List

import requests

__author__ = 'Dasha Herrmannova'
__email__ = 'dasha.herrmannova@open.ac.uk'


class CrossrefResolver(object):

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._endpoint = 'https://api.crossref.org/'

    def _format_date(self, date_parts: List[int]) -> str:
        """Format Crossref date
        
        :param date_parts: date parts
        :type date_parts: List[int]
        :return: date in the following format: %Y-%m-%d
        :rtype: str
        """
        if len(date_parts) == 3:
            return datetime.strftime(datetime.strptime(
                '-'.join([str(x) for x in date_parts]), '%Y-%m-%d'
            ), '%Y-%m-%d')  
        else:
            return '-'.join([str(x) for x in date_parts])

    def _process_response(self, response: str) -> Optional[Dict[str, str]]:
        """Process response received from Crossref
        
        :param response: response received from Crossref as string
        :type response: str
        :return: None in case response is not valid JSON, dictionary with the
                 following three fields otherwise: 'doi', 'title', 'published'
        :rtype: Optional[Dict[str, str]]
        """
        try:
            result = json.loads(response)
        except ValueError:
            return None
        result = result['message']
        published = result['issued']['date-parts'][0]
        if len(published) == 2:
            published.append(1)
        return {
            'doi': result['DOI'],
            'title': result['title'][0],
            'published': self._format_date(published),
        }

    def get_by_doi(self, doi: str) -> Optional[Dict[str, str]]:
        """Get metadata from Crossref by DOI
        
        :param doi: DOI to get metadata for 
        :type doi: str
        :return: None in case response is not valid JSON, dictionary with the
                 following three fields otherwise: 'doi', 'title', 'published'
        :rtype: Optional[Dict[str, str]]
        """

        request = urlparse.urljoin(self._endpoint, 'works/' + doi)
        self._logger.debug('CrossRef query: {}'.format(request))
        response = requests.get(request)
        if response.status_code == requests.codes.ok and len(response.text) > 0:
            return self._process_response(response.text)
        self._logger.debug('DOI {} not found'.format(doi))
        return None
