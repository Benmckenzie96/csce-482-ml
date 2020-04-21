from google.cloud import datastore
from org import Org
from org_dataset import OrgDataset

client = datastore.Client()

def get_org_dataset():
    """Fetches organizational data from google cloud
    datastore and creates an OrgDataset instance containing
    the data fetched.

    Returns:
        An OrgDataset instance containg the data fetched.
    """
    query = client.query(kind='organization')
    results = list(query.fetch())
    orgs = []
    for result in results:
        orgs.append(Org(result['orgId'], result['orgName'], result['orgPurpose']))
    od = OrgDataset()
    od.add_orgs(orgs)
    return od

def get_account_liked_orgs(account_id):
    """Fetches the ids of the orgs that the user
    is interested in.

    Args:
        account_id (str): The id of the user to
            fetch interested orgs for.

    Returns:
        A python list of strings. Each entry is an
        id of an organization the user is interested
        in.
    """
    query = client.query(kind='account')
    query.add_filter('userId', '=', account_id)
    results = list(query.fetch())
    if len(results) != 1:
        raise ValueError('More or less than 1 user returned.'
            ' Something went wrong.')
    return results[0]['userInterestOrgsId']

def get_account_disliked_orgs(account_id):
    """Fetches the ids of the orgs that the user
    is NOT interested in.

    Args:
        account_id (str): The id of the user to
            fetch not interested orgs for.

    Returns:
        A python list of strings. Each entry is an
        id of an organization the user is not interested
        in.
    """
    query = client.query(kind='account')
    query.add_filter('userId', '=', account_id)
    results = list(query.fetch())
    if len(results) != 1:
        raise ValueError('More or less than 1 user returned.'
            ' Something went wrong.')
    return results[0]['userDislikeOrgsId']
