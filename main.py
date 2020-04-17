from gcd_utils import get_org_dataset, get_account_liked_orgs
from vector_space import VectorSpace
from org_dataset import OrgDataset
import numpy as np
from org_recommender import OrgRecommender

# dataset = OrgDataset.load_instance('./orgs.pkl')
# vs = VectorSpace.load_instance('./orgs_vs.pkl')
# vecs = vs.transform(dataset.get_org_descriptions(range(70, 80)))
# print(vs.get_nearest_orgs(np.mean(vecs, axis=0), 20))
dataset = OrgDataset.load_instance('./orgs.pkl')
vs = VectorSpace.load_instance('./orgs_vs.pkl')
org_rec = OrgRecommender(dataset, vs)
id = '334614c0-7f55-11ea-b1bc-2f9730f51173'
print(org_rec.recommend_orgs(id, 5))
