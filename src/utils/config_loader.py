
import json
from typing import Dict, Any
from os.path import dirname, abspath, join

__author__ = 'Dasha Herrmannova'
__email__ = 'dasha.herrmannova@open.ac.uk'


class ConfigLoader(object):

    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load app config.
        
        :return: dictionary containing app config
        :rtype: Dict[str, Any]
        """
        root = dirname(dirname(dirname(abspath(__file__))))
        with open(join(root, 'config.json')) as fp:
            return json.load(fp)
