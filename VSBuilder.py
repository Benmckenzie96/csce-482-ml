"""
Created By: Cameron Przybylski

Creation Date: March 25, 2020

Purpose: This file contains functionality to create a
    vector space. The sklearn library is the primary
    driver of the functionality contained here.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from keras.utils import plot_model
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import tensorflow
from orgdata_json_utils import org_json_to_dictionary

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def import_data(filename):
    f = open(filename)
    jsonFile = json.load(f)
    return jsonFile

def get_names(jsonData):
    data = []
    for i in jsonData:
        data.append(i["name"])
    return data

def get_descs(jsonData):
    data = []
    for i in jsonData:
        data.append(i["desc"])
    return data

def get_vectors(names, descs):
    combinedData = {}
    for i in range(len(names)):
        combinedData[names[i]] = descs[i]
    return combinedData

def bag_of_words(vector):
    newVector = vector.split()
    dct = {}
    for i in newVector:
        if i in dct.keys():
            dct[i] = dct[i] + 1
        else:
            dct[i] = 1
    return dct


def data_to_list(dct):
    doc = []
    for key in dct.keys():
        tempStr = key + dct[key]
        doc.append(tempStr)
    return doc

def build_vector_space(combinedData):
    documents = data_to_list(combinedData)
    count_vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix_docs = count_vectorizer.fit_transform(documents)
    return (sparse_matrix_docs, count_vectorizer)


def k_means(matrix, vectorizer, k, show):
    km = KMeans(n_clusters=k)
    km.fit(matrix)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()

    if show == "print":
        for i in range(k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    return km


def elbow_method(matrix, vectorizer):
    sumOfSquareDistances = []
    distortions = []
    K = range(1,3)
    for k in K:
        km = KMeans(n_clusters=k).fit(matrix)
        km = km.fit(matrix)
        sumOfSquareDistances.append(km.inertia_)
        #distortions.append(sum(np.min(cdist(matrix, km.cluster_centers_, 'euclidean'),axis=1)) / matrix.shape[0])
    #print(sumOfSquareDistances)
    plt.plot(K, sumOfSquareDistances)
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    #plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    #plt.show()
    plt.savefig("Elbow_method_for_optimal_k.png")

def get_random_recommedations(data, n):
    recommendations = []
    for i in range(n):
        recommendation = random.choice(list(data.keys()))
        recommendations.append(recommendation)
    return recommendations

def closest_docs(combinedData, words, n):
    documents = data_to_list(combinedData)
    documents.append(words)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    docMatrix = tfidf_vectorizer.fit_transform(documents)

    inputIndex = documents.index(words)

    vals = cosine_similarity(docMatrix)
    np.fill_diagonal(vals, np.nan)

    results = []
    #tempVals = []
    #for i in vals:
        #tempVals.append(i)


    for i in range(n):
        inputIndex = documents.index(words)
        #vals = cosine_similarity(docMatrix)
        #np.fill_diagonal(vals, np.nan)
        resultIndex = np.nanargmax(vals[inputIndex])
        results.append(documents[resultIndex])
        np.delete(vals, resultIndex)
        #print("Working")

    #return results
    #resultIndex = np.nanargmax(vals[inputIndex])
    return results #documents[resultIndex]






if __name__ == "__main__":
    #data = org_json_to_dictionary('org_data.json')
    #print(data_to_list(data)[0])
    #data = org_json_to_dictionary("org_data.json")
    data = import_data("org_data.json")
    names = get_names(data)
    descriptions = get_descs(data)
    combinedData = get_vectors(names, descriptions)
    docMatrix, vectorizer = build_vector_space(combinedData)
    plot_model(docMatrix, to_file='model.png')
    # #k_means(docMatrix, vectorizer, 5, "print")
    # elbow_method(docMatrix, vectorizer)
    # #recommendations = get_random_recommedations(combinedData, 10)
    # #print(recommendations)
    # #words = "Computer Science and Engineering"
    # #docs = closest_docs(combinedData, words, 5)
    # #print(docs)