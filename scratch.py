"""
Created By: Ben McKenzie

Creation Date: March 31, 2020

Purpose: This is nothing more than a scratch file.
"""
from orgdata_json_utils import org_json_to_list, org_json_to_dictionary
from vector_space import VectorSpace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = org_json_to_dictionary('./data/org_data.json')
v = VectorSpace(data, max_features=100)
d_list = org_json_to_list(data)
index = v.get_nearest_orgs(d_list[0:1], 8)
print(type(index))
