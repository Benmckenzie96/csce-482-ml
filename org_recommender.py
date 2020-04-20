from gcd_utils import get_account_liked_orgs
import numpy as np

class OrgRecommender:
    """This class encapsulates the functionality required
    to recommend new organizations to a given user, based
    off of the organizations that they are interested in.

    Attributes:
        dataset (OrgDataset): an OrgDataset instance. This
            dataset must contain the same data that is stored
            online in order for predictions to have maximum
            accuracy
        vs (VectorSpace): an already initialized VectorSpace
            instance. This instance must be initialized with the
            same OrgDataset contained in dataset attr.
    """

    def __init__(self, org_dataset, org_vector_space):
        """Initializes an OrgRecommender with the given
        arguments. See class attribute documentation above
        for more info.
        """
        self.dataset = org_dataset
        self.vs = org_vector_space

    def recommend_orgs(self, user_id, num_orgs):
        """Uses vs attribute to provide recommended orgs
        based off of the user's liked orgs. Orgs are recommended
        based off of cosine proximity to the centroid of the
        user's liked orgs.

        Args:
            user_id (str): The id of the user you wish to
                generate recommendations for.
            num_orgs (int): the number of new orgs to
                recommend.
            random_org_count (int, optional): In the case that
                the supplied user doesn't have any liked orgs,
                this is the number of random orgs that will be used
                to provide recommendations.

        Returns:
            A list of organization ids. These are the ids of the
            recommended orgs for the user.
        """
        liked_orgs = get_account_liked_orgs(user_id)
        if len(liked_orgs) == 0:
            return self.dataset.get_random_org_ids(num_orgs)
        org_descs = self.dataset.get_orgs_by_id(liked_orgs, only_desc=True)
        num_to_drop = len(org_descs)
        org_vecs = self.vs.transform(org_descs)
        centroid = np.mean(org_vecs, axis=0)
        num_to_fetch = num_orgs + num_to_drop
        df = self.vs.get_nearest_orgs(centroid, num_to_fetch)
        return df.loc[~df['orgId'].isin(liked_orgs)]['orgId'].to_numpy()
