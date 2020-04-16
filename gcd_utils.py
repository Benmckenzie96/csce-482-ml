from google.cloud import datastore
from org import Org
from org_dataset import OrgDataset

def get_org_dataset():
    GOOGLE_APPLICATION_CREDENTIALS = '/Users/benmckenzie/programming-workspace/482-keys/local-cred.json'
    client = datastore.Client.from_service_account_json(GOOGLE_APPLICATION_CREDENTIALS)
    query = client.query(kind='organization')
    results = list(query.fetch())
    orgs = []
    for result in results:
        orgs.append(Org(result['orgId'], result['orgName'], result['orgPurpose']))
    od = OrgDataset()
    od.add_orgs(orgs)
    return od

# # getting users
# query = client.query(kind='account')
# print(list(query.fetch()))
