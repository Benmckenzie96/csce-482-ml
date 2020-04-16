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

    def get_org_descriptions(self):
        """Returns a numpy array of strings. Each entry
        is an 'Org' description. The description consists of
        the 'Org' name concatenated with the 'Org' purpose.
        """
        return self.dataframe['orgPurpose'].to_numpy()

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
