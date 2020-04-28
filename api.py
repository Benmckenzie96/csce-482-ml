from fastapi import FastAPI
from vector_space import VectorSpace
from org_dataset import OrgDataset
from org_recommender import OrgRecommender
from clusterer import Clusterer
from keyword_finder import KeywordFinder
from keyword_matcher import KeywordMatcher
from gcd_utils import get_account_liked_tags

app = FastAPI()
dataset = OrgDataset.load_instance('./orgs.pkl')
vs = VectorSpace.load_instance('./test_vs.pkl')
recommender = OrgRecommender(dataset, vs)

c = Clusterer(dataset, vs, 20)
kw_finder = KeywordFinder(dataset, vs)
matcher = KeywordMatcher(c, kw_finder, vs.data_centroid)

@app.get('/get_init_recs/')
async def get_init_recs(userId: str, numOrgs: int):
    keywords = get_account_liked_tags(userId)
    centroid = matcher.get_kw_centroid(keywords)
    orgids = recommender.centroid_recommend(centroid, numOrgs)
    return_arr = []
    for id in orgids:
        entry = {'orgId': id}
        return_arr.append(entry)
    return return_arr
"""Example get request for api on local host:

http://127.0.0.1:8000/get_recommendations/?userId=334614c0-7f55-11ea-b1bc-2f9730f51173&numOrgs=2
"""

@app.get('/get_recommendations/')
async def get_recommendations(userId: str, numOrgs: int):
    random_id = dataset.get_random_org_ids(1)
    orgids = recommender.recommend_orgs(userId, numOrgs)
    return_arr = [{'orgId': random_id[0]}]
    for id in orgids:
        entry = {'orgId': id}
        return_arr.append(entry)
    return return_arr

#test_id = '2bd2e4a0-85ce-11ea-9f05-e3bd91f1b63a'
