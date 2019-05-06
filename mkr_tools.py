import requests
import json
from tqdm import tqdm

def get(url):
    return json.loads(requests.get(url).text)

def fetch_all_cdps():
    return get('https://mkr.tools/api/v1/cdps')

def fetch_data_for_cdp(cdp, endpoint=''):
    return get('https://mkr.tools/api/v1/cdp/{}/{}'.format(cdp, endpoint))

def fetch_all():
    cdps = fetch_all_cdps()
    for cdp in tqdm(cdps):
        cdp['actions'] = fetch_data_for_cdp(cdp['id'], 'actions')
        cdp['history'] = fetch_data_for_cdp(cdp['id'], 'history')
    return cdps

# data = fetch_all() # takes ~10 hours
# cdps = fetch_all_cdps() # quick