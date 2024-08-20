import numpy as np
import pandas as pd
import logging
from libraries.pipeconfig import RatingScaleEnum


class MatrixLoader:
    """
    Joseph Weibel
    Class to load long and wide matrices efficiently by caching them.
    """
    CONST_INDEXID = "movieId"

    loaded_matrices = {}

    def loadWide(self, file):
        """
        Joseph Weibel
        Loads the requested CSV file and sets movieId as index.
        :param str file:
            filename of the file to load. Ensure you use the exact same path to make optimal use of the cache.
            Store the filename as a constant to ensure it is always the same.
        :return: pd.DataFrame the requested matrix loaded from the filesystem or from the cache
        :rtype: pd.DataFrame
        """

        if file in MatrixLoader.loaded_matrices:
            return MatrixLoader.loaded_matrices[file].copy()

        matrix = pd.read_csv(file, engine="pyarrow").set_index(self.CONST_INDEXID)
        MatrixLoader.loaded_matrices[file] = matrix
        return matrix.copy()

    def loadLong(self, file):
        """
        Joseph Weibel
        Loads the requested CSV file.
        :param str file:
            filename of the file to load. Ensure you use the exact same path to make optimal use of the cache.
            Store the filename as a constant to ensure it is always the same.
        :return: pd.DataFrame the requested matrix loaded from the filesystem or from the cache
        :rtype: pd.DataFrame
        """

        if file in MatrixLoader.loaded_matrices:
            return MatrixLoader.loaded_matrices[file].copy()

        matrix = pd.read_csv(file, engine="pyarrow")
        MatrixLoader.loaded_matrices[file] = matrix
        return matrix.copy()

class MatrixGenerator(object):
    """
    Thomas Mandelz
    Class to generate feature matrices for different types of metadata of movies
    """
    # Constants for Filepaths of Metadata, DO NOT CHANGE except if you know what you're doing
    CONST_MetaDataCSVPath = r"./data/movies_meta.csv"
    CONST_GenresCSVPath = r"./data/movies_genres.csv"
    CONST_ActorsCSVPath = r"./data/movies_casts.csv"
    CONST_DirectorsCSVPath = r"./data/movies_crew.csv"
    CONST_TfidfTokensCSVPath = r"./data/movies_tfidf.csv"
    CONST_BertEmbeddingsCSVPath = r"./data/movies_bert.csv"
    CONST_SBertEmbeddingsCSVPath = r"./data/movies_sbert.csv"
    CONST_RandomCSVPath = r"./data/movies_random.csv"

    CONST_UserItem_BIN = r"./data/userItem_bin.csv"
    CONST_UserItem_TERT = r"./data/userItem_tert.csv"
    CONST_UserItem_STAND = r"./data/userItem_stand.csv"

    CONST_KEY_METADATA = 'metadata'
    CONST_KEY_GENRES = 'genres'
    CONST_KEY_ACTORS = 'actors'
    CONST_KEY_DIRECTORS = 'crew'
    CONST_KEY_TFIDF = 'tfidf'
    CONST_KEY_BERT = 'bert'
    CONST_KEY_SBERT = 'sbert'
    CONST_KEY_RANDOM = 'random'

    def __init__(
        self,
        metadata: bool = False,
        genres: bool = False,
        actors: bool = False,
        directors: bool = False,
        tfidfTokens: bool = False,
        bertEmbeddings: bool = False,
        sbertEmbeddings: bool = False,
        random: bool = False,
    ) -> None:
        """
        Thomas Mandelz
        init method for class to generate featurematrix for different types of metadata of movies
        :param bool|str metadata: bool if metadata should be included in returned matrix or filepath as str to load that matrix
        :param bool|str genres: bool if genres should be included in returned matrix or filepath as str to load that matrix
        :param bool|str actors: bool if actors should be included in returned matrix or filepath as str to load that matrix
        :param bool|str directors: bool if directors should be included in returned matrix or filepath as str to load that matrix
        :param bool|str tfidfTokens: bool if TFIDF tokens should be included in returned matrix or filepath as str to load that matrix
        :param bool|str bertEmbeddings: bool if BERT sentence embeddings should be included in returned matrix or filepath as str to load that matrix
        :param bool|str random: bool if random features should be included in returned matrix or filepath as str to load that matrix
        """
        self.metadata = metadata
        self.genres = genres
        self.actors = actors
        self.directors = directors
        self.tfidfTokens = tfidfTokens
        self.bertEmbeddings = bertEmbeddings
        self.sbertEmbeddings = sbertEmbeddings
        self.random = random

        self.loader = MatrixLoader()

    def createFeatureMatrix(self) -> pd.DataFrame:
        """
        Thomas Mandelz
        Function which concatenates the chosen metadata and returns a single dataframe with a multiindex
        :return: pd.DataFrame as a Featurematrix
        :rtype: pd.DataFrame
        """
        dictDf = {}
        def loadIfNeeded(flag, key, path):
            if flag:
                dictDf[key] = self.loader.loadWide(flag if isinstance(flag, str) else path)

        # add subsets of all metadata to dictionary for later concatenation
        try:
            loadIfNeeded(self.metadata, self.CONST_KEY_METADATA, self.CONST_MetaDataCSVPath)
            loadIfNeeded(self.genres, self.CONST_KEY_GENRES, self.CONST_GenresCSVPath)
            loadIfNeeded(self.actors, self.CONST_KEY_ACTORS, self.CONST_ActorsCSVPath)
            loadIfNeeded(self.directors, self.CONST_KEY_DIRECTORS, self.CONST_DirectorsCSVPath)
            loadIfNeeded(self.tfidfTokens, self.CONST_KEY_TFIDF, self.CONST_TfidfTokensCSVPath)
            loadIfNeeded(self.bertEmbeddings, self.CONST_KEY_BERT, self.CONST_BertEmbeddingsCSVPath)
            loadIfNeeded(self.sbertEmbeddings, self.CONST_KEY_SBERT, self.CONST_SBertEmbeddingsCSVPath)
            loadIfNeeded(self.random, self.CONST_KEY_RANDOM, self.CONST_RandomCSVPath)
        except Exception as e:
            logging.error(
                f"Error while reading Metadata CSV's \n Exception Message: {e}"
            )
            raise e

        try:
            # Concatenate the dataframes in the dictionary and add the keys as a multiindex
            # https://stackoverflow.com/questions/23600582/concatenate-pandas-columns-under-new-multi-index-level
            matrix = pd.concat(dictDf.values(), axis=1, keys=dictDf.keys())
        except Exception as e:
            logging.error(
                f"Empty Dataframe was generated! \n Exception Message: {e}")
            raise e

        return matrix

    def createRatingMatrix(
        self,
        ratingScale: RatingScaleEnum,
    ) -> pd.DataFrame:
        """
        Thomas Mandelz
        Function which reads ratings as a csv
        :param RatingScaleEnum ratingScale: enum to read the ratingscale
        :return: pd.DataFrame as a Ratingsmatrix
        :rtype: pd.DataFrame
        """
        try:
            if ratingScale == RatingScaleEnum.STANDARDISED:
                return self.loader.loadWide(self.CONST_UserItem_STAND)
            if ratingScale == RatingScaleEnum.TERTIARY:
                return self.loader.loadWide(self.CONST_UserItem_TERT)
            if ratingScale == RatingScaleEnum.BINARY:
                return self.loader.loadWide(self.CONST_UserItem_BIN)
        except Exception as e:
            logging.error(
                f"Error while reading Rating CSV's \n Exception Message: {e}")
            raise e
