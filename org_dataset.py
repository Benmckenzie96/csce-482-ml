import pandas as pd
from org import Org
import pickle

class OrgDataset:
    """This class contains functionality to create
    a local database of organizations. The database is
    a pandas dataframe under the hood.

    Attributes:
        attributes (list): a list of strings. Each entry
            is a column label for the dataframe.
        dataframe (DataFrame): the dataframe containing
            all of the organization data. You shouldn't have
            to directly interact with this. Accessing the dataframe
            should be done using the functions of this class.
    """

    def __init__(self):
        """Initializes an OrgDataset object with an
        empty dataframe to store data. The dataframe
        has the same columns as the values returned by
        'attribute_labels' function found in 'Org' class.
        """
        self.attributes = Org.attribute_labels()
        self.dataframe = pd.DataFrame(columns=self.attributes)

    def add_orgs(self, orgs):
        """Adds a list of 'Org' objects to the database.

        Args:
            orgs (list): a list of Org objects
        """
        data = []
        for org in orgs:
            data.append(org.data())
        new_orgs = pd.DataFrame(data=data, columns=self.attributes)
        self.dataframe = pd.concat([self.dataframe, new_orgs])

    def get_orgs_by_indices(self, indices):
        """Returns the rows of the dataframe with
        indices equal to the supplied indices.

        Args:
            indices (list): A python list of integers.

        Returns:
            A subset of the dataframe containing all
            organizations stored. All fields of the
            dataframe are returned.
        """
        return self.dataframe.iloc[indices]

    def get_org_descriptions(self, indices=None):
        """
        Args:
            indices (list): an optional list of integers
                representing which rows of the orgs dataframe
                should have descriptions returned. If no value is
                provided, all org descriptions are returned.

        Returns:
            a numpy array of strings. Each entry
            is an 'Org' description. The description consists of
            the 'Org' name concatenated with the 'Org' purpose.
        """
        if indices is not None:
            df = self.get_orgs_by_indices(indices)
        else:
            df = self.dataframe
        return df['orgPurpose'].to_numpy()

    def get_orgs_by_id(self, ids, only_desc=False):
        """Gets the rows of the database dataframe with
        orgId equal to the supplied values.

        Args:
            ids (list): a list (or np array) of strings
                where each entry is an organization id.
            only_desc (bool, optional): defaults to False.
                if set to True, rather than dataframe rows
                being returned, only a list containing each
                organization's description will be returned.

        Returns:
            One of the following depending on the value for
            only_desc:
                False: a dataframe containing the orgs with ids
                    equal to ids.
                True: an array containing only the organizations'
                    description strings.
        """
        df = self.dataframe.loc[self.dataframe['orgId'].isin(ids)]
        if only_desc:
            return df['orgPurpose'].to_numpy()
        else:
            return df

    def save_instance(self, destination):
        """Saves OrgDataset instance to a pickle file
        in the specified destination.

        Args:
            destination (str): The destination to save
                This OrgDataset instance to.
        """
        with open(destination, 'wb') as f:
            pickle.dump(self, f)

    def load_instance(location):
        with open(location, 'rb') as f:
            return pickle.load(f)
