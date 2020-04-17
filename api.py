from fastapi import FastAPI
from vector_space import VectorSpace
from org_dataset import OrgDataset
from org_recommender import OrgRecommender

app = FastAPI()
dataset = OrgDataset.load_instance('./orgs.pkl')
vs = VectorSpace.load_instance('./orgs_vs.pkl')
recommender = OrgRecommender(dataset, vs)

"""Example get request for api on local host:

http://127.0.0.1:8000/get_recommendations/?userId=334614c0-7f55-11ea-b1bc-2f9730f51173&numOrgs=2
"""

@app.get('/get_recommendations/')
async def get_recommendations(userId: str, numOrgs: int):
    orgids = recommender.recommend_orgs(userId, numOrgs)
    return_arr = []
    for id in orgids:
        entry = {'org': id}
        return_arr.append(entry)
    return return_arr
