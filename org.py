class Org:
    """This class defines an Org object.

    Attributes:
        org_id (str): The id of the org from google cloud
            datastore.
        org_name (str): The name of the org from google cloud
            datastore.
        org_purpose (str): The name concatenated with the purpose
            of the organization. The reason the values are concatenated
            is that some orgs have not purposes listed.
    """

    def __init__(self, org_id, org_name, org_purpose):
        """Initializes an Org instance.
        """
        self.org_id = org_id
        self.org_name = org_name.strip()
        self.org_purpose = self.org_name + ' ' + org_purpose.strip()

    def info(self):
        """returns a string containing instance's values
        for each of the features.
        """
        return('id: {}, name: {}, desc: {}'.format(self.org_id, self.org_name, self.org_purpose))

    def data(self):
        """returns a list of data values for instance.
        """
        return [self.org_id, self.org_name, self.org_purpose]

    def attribute_labels():
        """Simply returns the names of each feature
        that org objects have.
        """
        return ['orgId', 'orgName', 'orgPurpose']
