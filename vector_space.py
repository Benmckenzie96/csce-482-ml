"""
Created By: Ben McKenzie

Creation Date: March 31, 2020

Purpose: Define a class that contains vector space
    functionality. This class is designed to work with
    organizational json data. It is NOT safe to assume that
    the functionality in this class will generalize to tasks
    outside of this domain.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from orgdata_json_utils import org_json_to_list
import numpy as np

class VectorSpace:

    def __init__(self, data, analyzer='word', stop_words='english',
            ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None):
        """Initializes a VectorSpace instance. The vector space is fit on
        the data provided by 'data' argument.

        Args:
            data (dictionary): a dictionary containing organizational
                data. The keys for this dictionary should be organization names,
                and the values should be the corresponding descriptions.
            analyzer (str, {‘word’, ‘char’, ‘char_wb’}): Whether the
                feature should be made of word or character n-grams.
                Option ‘char_wb’ creates character n-grams only from
                text inside word boundaries; n-grams at the edges of
                words are padded with space.
            stop_words (str {‘english’}, list, or None): If a string, it
                is passed to _check_stop_list and the appropriate stop list is
                returned. ‘english’ is currently the only supported string value.
                If a list, that list is assumed to contain stop words, all of which
                will be removed from the resulting tokens.
                Only applies if analyzer == 'word'.
            ngram_range (tuple (min_n, max_n)): The lower and upper boundary
                of the range of n-values for different n-grams to be extracted.
                All values of n such that min_n <= n <= max_n will be used.
                For example an ngram_range of (1, 1) means only unigrams,
                (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams.
            max_df (float in range [0.0, 1.0] or int): When building the vocabulary,
                ignore terms that have a document frequency strictly higher than the
                given threshold (corpus-specific stop words). If float, the parameter
                represents a proportion of documents, integer absolute counts.
            min_df (float in range [0.0, 1.0] or int): When building the vocabulary,
                ignore terms that have a document frequency strictly lower than the
                given threshold. This value is also called cut-off in the literature.
                If float, the parameter represents a proportion of documents,
                integer absolute counts.
            max_features (int or None): If not None, build a vocabulary that only
                considers the top max_features ordered by term frequency across
                the corpus.

            Returns:
                A VectorSpace instance fitted on the data provided
                    by 'data'.
        """
        self.data = data
        self.analyzer = analyzer
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, stop_words=stop_words,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)\
            .fit(org_json_to_list(self.data))
        self._transformed_data = self.vectorizer.transform(org_json_to_list(self.data))

    def info(self):
        """Returns a string containing the information of
        of the VectorSpace instance.
        """
        s = 'analyzer: {}, stop_words: {}, ngram_range: {},'\
            ' max_df: {}, min_df: {}, max_features: {}'\
            .format(self.analyzer, self.stop_words, self.ngram_range,
                self.max_df, self.min_df, self.max_features)
        return s

    def get_stop_words(self):
        """Returns the effective stop word list
        of VectorSpace instance. Note that this list
        is not the stop words you provide in the __init__
        method, but rather the effective stopwords obtained
        based off of min and max df.

        stop words are returned as a set.
        """
        return self.vectorizer.stop_words_

    def get_vocabulary(self):
        """Returns a dictionary of mapping of terms
        to feature indices.
        """
        return self.vectorizer.vocabulary_

    def transform(self, input, to_numpy=False):
        """Transforms input into tfidf embedded vectors.
        Vector embeddings of input are returned as a matrix.

        Args:
            input (iterable): a list like object containing
                strings, where each string represents a document
                that you wish to embed.
            to_numpy (bool): indicates whether embeddings should
                be returned as a numpy matrix or not. Default value
                is False.

        Returns:
            If to_numpy is False, a scipy csr matrix is returned.
                If to_numpy is True, document embeddings are returned
                as a numpy matrix.
        """
        embeddings = self.vectorizer.transform(input)
        if to_numpy:
            return embeddings.todense()
        else:
            return embeddings

    def get_nearest_orgs(self, input_vector, k=1):
        """Gets the nearest organizations stored in
        instance's data attribute. The keys containing
        the organization names are returned.

        Args:
            input_vector (scipy csr vector): a vector from the
                same vector space that this VectorSpace instance
                represents.
            k (int): The number of closest orgs to fetch.

        Returns:
            A python list containing the organization keys
                that are closest to input_org in terms of
                cosine cosine similarity
        """
        sims = cosine_similarity(input_vector, self._transformed_data)[0]
        indices = np.argpartition(sims, -(k))[-(k):]
        input_index = np.argpartition(sims, -1)[-1:]
        indices = list(indices)
        dict_keys = []
        for index in indices:
            key = list(self.data.keys())[index]
            dict_keys.append(key)
        return dict_keys

    # def get_nearest_orgs(self, input_org, k=1):
    #     """Gets the nearest organizations stored in
    #     instance's data attribute. The keys containing
    #     the organization names are returned.
    #
    #     Args:
    #         input_org (list of dimension [[1]]): A list
    #             containing a single string. The string is
    #             the organization you wish to fetch nearest
    #             neighbors for.
    #         k (int): The number of closest orgs to fetch.
    #
    #     Returns:
    #         A python list containing the organization keys
    #             that are closest to input_org in terms of
    #             cosine cosine similarity
    #     """
    #     if len(input_org) != 1:
    #         raise ValueError('Currently only single orgs'
    #          ' are supported as input.')
    #     print(input_org)
    #     print('\n')
    #     vectorized_org = self.transform(input_org)
    #     sims = cosine_similarity(vectorized_org, self._transformed_data)[0]
    #     # add 1 to k to factor in self sim of input.
    #     indices = np.argpartition(sims, -(k + 1))[-(k + 1):]
    #     # This code is to remove the organization
    #     # itself from consideration. input_org will
    #     # always be most similar to itself.
    #     input_index = np.argpartition(sims, -1)[-1:]
    #     indices = list(indices)
    #     indices.remove(input_index[0])
    #     dict_keys = []
    #     for index in indices:
    #         key = list(self.data.keys())[index]
    #         dict_keys.append(key)
    #         print(self.data[key])
    #         print('\n')
    #     return dict_keys
