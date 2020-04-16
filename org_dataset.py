import pandas as pd
from org import Org

class OrgDataset:

    def __init__(self):
        self.attributes = Org.attribute_labels()
        self.dataframe = pd.DataFrame(columns=self.attributes)

    def add_orgs(self, orgs):
        data = []
        for org in orgs:
            data.append(org.data())
        new_orgs = pd.DataFrame(data=data, columns=self.attributes)
        self.dataframe = pd.concat([self.dataframe, new_orgs])

    def get_org_descriptions(self):
        return self.dataframe['orgPurpose'].to_numpy()

    def get_orgs_by_indices(self, indices):
        return self.dataframe.iloc[indices]
