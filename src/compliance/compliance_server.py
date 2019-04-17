
import json
import logging

from flask import Flask, request, Response

from src.compliance.compliance_checker import ComplianceChecker

app = Flask(__name__)


@app.route('/check_compliance', methods=['GET', 'POST'])
def check_compliance():
    logger = logging.getLogger(__name__)

    with open('config.json') as fp:
        cfg = json.load(fp)
    max_request_size = cfg['api']['max_request_size']
    logger.info('Max request size: {}'.format(max_request_size))
    
    cc = ComplianceChecker(
        cfg['core']['api_key'], cfg['core']['max_query_size']
    )

    result = None
    status = 200
    if request.method == 'GET':
        logger.info('Received a GET request')
        doi = request.args.get('doi')
        result = cc.compliance_status([doi])
    else:
        dois = request.get_json()
        logger.info('Received {} dois'.format(len(dois)))
        if len(dois) > max_request_size:
            result = 'Request too large (max size: {})'.format(max_request_size)
            status = 400
        else:
            result = cc.compliance_status(dois)
    return Response(
        json.dumps(result), status=status, mimetype='application/json'
    )


def run_server():
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=8124, threaded=True)
