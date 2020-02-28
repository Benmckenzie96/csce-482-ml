"""This file contains various utility functions related
to the loading and manipulation of json data. The data is assumed
to have the following schema:
[
    {
        "name" : string,
        "description" : string,
        "people" : [
            array of strings
        ]
    }
]

Created By: Cameron Przybylski

Creation Date: 2/24/20
"""
import json

def __org_import_data(filename):
    """This function loads a json file stored
    in the location specified by filename into
    a python list.

    Args:
        filename (string): The file location of the
            json file you wish to load.

    Returns:
        A python list containing the json data.
    """
    with open(filename, "r") as f:
        jsonFile = json.load(f)
    return jsonFile

def __org_get_names(jsonData):
    """This function returns all of the values for
    the "name" key as specified in the schema at the top
    of this file.

    Args:
        jsonData (list): a python list containing json data.
            This list can be obtained by calling 'import_data'.

    Returns:
        A python list of strings where each element is the
        name of an organization.
    """
    data = []
    for i in jsonData:
        data.append(i["name"])
    return data


def __org_get_descs(jsonData):
    """This function returns all of the values for
    the "description" key as specified in the schema at the top
    of this file.

    Args:
        jsonData (list): a python list containing json data.
            This list can be obtained by calling 'import_data'.

    Returns:
        A python list of strings where each element is the
        description of an organization.
    """
    data = []
    for i in jsonData:
        data.append(i["desc"])
    return data


def __org_get_vectors(names, descs):
    """This function combines the name and description fields
    of json data into a dictionary.

    Args:
        names (list): python list of organization names.
            this list can be obtained by calling 'import_data'
            followed by 'get_names' on the value returned.
        descs (list): python list of organization descriptions.
            this list can be obtained by calling 'import_data'
            followed by 'get_descs' on the returned result.

    Returns:
        A python dictionary. Each key in the dictionary is
        an organization name, each value in the dictionary is
        the corresponding organization description.
    """
    combinedData = {}
    for i in range(len(names)):
        combinedData[names[i]] = descs[i]
    return combinedData

def org_json_to_dictionary(filename):
    """This function converts json data following the
    schema specified at the top of this file into a
    python dictionary. The "people" array of the data
    is omitted for each entry.

    Args:
        filename (string): The location of the organizational
            json data you wish to convert to a dictionary.

    Returns:
        A python dictionary. Each key in the dictionary is
        an organization name, each value in the dictionary is
        the corresponding organization description.
    """
    data = __org_import_data(filename)
    names = __org_get_names(data)
    descs = __org_get_descs(data)
    return __org_get_vectors(names, descs)

# Example use case:
if __name__ == '__main__':
    dict = org_json_to_dictionary("../data/org_data.json")
    print(dict)
