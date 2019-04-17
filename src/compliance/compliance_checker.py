
import json
import logging
from enum import Enum
from datetime import datetime
from typing import List, Dict, Any

from src.compliance.core_resolver import CoreResolver
from src.compliance.crossref_resolver import CrossrefResolver

__author__ = 'Dasha Herrmannova'
__email__ = 'dasha.herrmannova@open.ac.uk'


class Statuses(Enum):
    DOI_ERR = 'Invalid DOI'
    NONCOMPLIANT = 'Not compliant'
    COMPLIANT = 'Likely compliant'
    DATA_ERR = 'Missing data'


class ComplianceChecker(object):

    def __init__(self, core_api_key: str, core_max_query_size: int) -> None:
        """
        :param core_api_key: CORE API key, can be obtained from here: 
                             https://core.ac.uk/api-keys/register/
        :type core_api_key: str
        :param core_max_query_size: maximum size of one query, defaults to 100
        :type core_max_query_size: int
        :return: None
        :rtype: None
        """
        self._logger = logging.getLogger(__name__)
        self._crossref_resolver = CrossrefResolver()
        self._core_resolver = CoreResolver(core_api_key, core_max_query_size)

    def _compare_dates(self, published: str, deposited: str) -> str:
        """Compare publication and deposit dates and return compliance status
        as string
        
        :param published: date a paper was published, must be in the following
                          format: %Y-%m-%d
        :type published: str
        :param deposited: date the paper was deposited, must be in the following
                          format: %Y-%m-%d
        :type deposited: str
        :return: compliance status
        :rtype: str
        """
        try:
            published_date = datetime.strptime(published, '%Y-%m-%d')
            deposited_date = datetime.strptime(deposited, '%Y-%m-%d')
        except ValueError:
            return Statuses.DATA_ERR.value
        difference = (deposited_date - published_date).days
        if difference <= 90:
            return Statuses.COMPLIANT.value
        else:
            return Statuses.NONCOMPLIANT.value

    def _doi_valid(self, doi: Any) -> bool:
        """Check if the input is a nonempty string 
        
        :param doi: input to check
        :type doi: Any
        :return: True if input value is a nonempty string
        :rtype: bool
        """
        return doi and (type(doi) == str or type(doi) == bytes) and len(doi)

    def compliance_status(
            self, doi_list: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """Check deposit time lag and compliance status for a list of DOIs
        
        :param doi_list: list of DOIs to be checked
        :type doi_list: List[str]
        :return: [description]
        :rtype: Dict[str, Dict[str, str]]
        """
        valid = {doi: self._doi_valid(doi) for doi in doi_list}

        # CrossRef doesn't allow batching -- resolve each DOI separately
        cr_data = {
            doi: self._crossref_resolver.get_by_doi(doi) if v else None 
            for doi, v in valid.items()
        }

        # which DOIs to resolve -- avoid sending those that don't appear in CR
        core_doi_list = [doi for doi, v in valid.items() if v]
        core_data = self._core_resolver.get_by_dois(core_doi_list)

        # combine the maps
        results = {}
        for doi, doi_valid in valid.items():
            cr_doc = cr_data[doi] if doi in cr_data else None
            core_doc = core_data[doi] if doi in core_data else None
            if not doi_valid:
                results[doi] = {
                    'published': None,
                    'deposited': None, 
                    'status': Statuses.DOI_ERR.value
                }
            elif not cr_doc or not core_doc:
                results[doi] = {
                    'published': cr_doc['published'] if cr_doc else None,
                    'deposited': core_doc['deposited'] if core_doc else None, 
                    'status': Statuses.DATA_ERR.value
                }
            else:
                results[doi] = {
                    'published': cr_doc['published'] if cr_doc else None,
                    'deposited': core_doc['deposited'] if core_doc else None, 
                    'status': self._compare_dates(
                        cr_doc['published'], core_doc['deposited']
                    )
                }        
        return results
