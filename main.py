from gcd_utils import get_org_dataset
from vector_space import VectorSpace

dataset = get_org_dataset()
test_org1 = dataset.get_orgs_by_indices([0])
test_org2 = dataset.get_orgs_by_indices([37])
test = test_org1['orgPurpose'].to_numpy()[0] + test_org2['orgPurpose'].to_numpy()[0]
print(test)
vs = VectorSpace(dataset)
test_org_vec = vs.transform([test])
print(vs.get_nearest_orgs(test_org_vec, 2))
