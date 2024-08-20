# Imports python Libs
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import time
from tqdm import tqdm
import platform

# Imports our Libs
# from mypackage.mymodule import class_or_function
from libraries.matrix import MatrixGenerator
from libraries.pipeconfig import (
    CakeConfig,
    FeatureNormalizationEnum,
    SimilarityEnum,
    RatingScaleEnum,
)
from libraries.weights import WeightGenerator

isPDFGenerator = platform.system() == "Linux"

# Classes
class Cakeommender(object):
    """
    Thomas Mandelz
    Class to generate a cakeommender
    """

    def __init__(
        self,
        name: str,
        config: CakeConfig,
        matrixGenerator: MatrixGenerator,
        ratingsSplit: pd.DataFrame = None,
        loggingPath: str = r"./log/",
        nonNumericColumns: list = [
            "movieId",
            "tmdbId",
            "original_title",
            "overview",
            "release_date",
        ],
        verbose: bool = False,
        progress: bool = True
    ) -> None:
        """
        Thomas Mandelz
        init method for class to generate config object

        :param str name: Name of the Cakeommender instance
        :param CakeConfig config: cakeommender configuration object
        :param MatrixGenerator matrixGenerator: MatrixGenerator instance to generate feature matrix
        :param str loggingPath: Path to log filefolder
        :param list nonNumericColumns: list of column names which are not numeric and have to be excluded from non NLP recommendations
        :param bool verbose: Determines if logging is activated
        :param bool progress: Determines if progress should be shown
        """
        start = time.time()

        # Setting Attributes
        self.verbose = verbose
        self.name = name
        self.config = config

        # Setup Filelogger if wanted
        if self.verbose:
            logging.basicConfig(filename=f"{loggingPath}{self.name}.log",
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)

        with tqdm(total=8, disable=not progress, leave=not isPDFGenerator) as pbar:
            # generate featurematrix
            pbar.set_description('creating feature matrix')
            self.features = matrixGenerator.createFeatureMatrix()
            # Set numeric features from nonnumeric columnlist
            self.numericFeatures = self.features.loc[
                :, ~self.features.columns.get_level_values(1).isin(nonNumericColumns)
            ]
            pbar.update()

            # generate weights
            pbar.set_description('generating weights')
            self.weights = WeightGenerator().createWeights(
                self.config.weightsDistribution, self.numericFeatures, nonNumericColumns
            )
            pbar.update()

            # Standardise Features
            pbar.set_description('normalizing features')
            self.normalizeFeatures()
            # Fill NA's with a 0
            self.numericFeatures = self.numericFeatures.fillna(0)
            pbar.update()

            # Weights Scalar Multiplication
            pbar.set_description('weighting features')
            self.weighFeatures()
            pbar.update()

            # generate ratings
            if isinstance(ratingsSplit, pd.DataFrame) == True:
                pbar.set_description('generating rating split')
                self.ratings = ratingsSplit
            else:
                pbar.set_description('generating rating matrix')
                self.ratings = matrixGenerator.createRatingMatrix(
                    self.config.ratingScale
                )
            pbar.update()

            # get not rated movies and fill missing ratings for movies with zeros
            pbar.set_description('filling ratings')
            self.notRatedMovies = self.getNotRatedMovies()
            self.fillRatings()
            pbar.update()

            # create intersection of movies in feature and rating matrices
            pbar.set_description('aligning matrices')
            self.alignMatrices()
            pbar.update()

            # Calculate Userprofiles
            pbar.set_description('calculating user profiles')
            self.userProfiles = self.calcUserProfiles()
            pbar.update()
            pbar.set_description('cakeommender finished 🥳')

        end = time.time()
        if self.verbose:
            logging.info(f"{self}")
            logging.info(
                f"Initialisation of Recommender needed: {end - start} seconds!")

    def __str__(self) -> str:
        """
        Thomas Mandelz
        Overriden to-string method for logging
        """
        return f"{self.name}: \nRecommenders Features:\n{self.features} \nRecommenders NumericFeatures:\n{self.numericFeatures} \nRecommenders Ratings:\n{self.ratings} \nRecommenders Userprofiles:\n{self.userProfiles} \nRecommenders Config:\n{self.config}"

    def getNotRatedMovies(self):
        """
        Thomas Mandelz
        gets row difference for ratings and features
        """
        # get difference in index of rows
        dif = list(set(self.numericFeatures.index.values) -
                   set(self.ratings.index.values))
        return dif

    def fillRatings(self):
        """
        Thomas Mandelz
        adds ratings rows with zeros for these missing movies
        """
        self.ratings = self.ratings.reindex(self.ratings.index.tolist() + self.notRatedMovies).fillna(0)

    def alignMatrices(self):
        """
        Joseph Weibel
        ensures feature and rating matrices contain the same movies by creating an intersection between those two.
        """
        movieIds = self.numericFeatures.index.intersection(self.ratings.index)
        if len(movieIds) == 0:
            raise Exception('no common movies in numericFeatures and ratings')

        self.numericFeatures = self.numericFeatures.loc[movieIds]
        self.ratings = self.ratings.loc[movieIds]

    def normalizeFeatures(self):
        """
        Thomas Mandelz
        normalizes features of a dataframe with the defined normalisation approach
        """
        try:
            if self.config.featureNormalisation == FeatureNormalizationEnum.MINMAX:
                # Min Max Scaling (x - x.min /x.max - x.min), range = 0 to 1
                self.numericFeatures = (
                    self.numericFeatures - self.numericFeatures.min()
                ) / (self.numericFeatures.max() - self.numericFeatures.min())
            elif self.config.featureNormalisation == FeatureNormalizationEnum.ZSCORE:
                # z-score normalisation (x - x.mean / x.std), range = -infinite to + infinite
                self.numericFeatures = (
                    self.numericFeatures - self.numericFeatures.mean(skipna=True)
                ) / self.numericFeatures.std(skipna=True)
        except Exception as e:
            logging.error(
                f"Error while feature normalisation! \n Exception Message: {e}"
            )
            raise e

    def weighFeatures(self):
        """
        Thomas Mandelz
        scalarly multiplicates the feature weights with the featurematrix
        """
        try:
            self.numericFeatures = self.numericFeatures * self.weights.T
        except Exception as e:
            logging.error(
                f"Error while feature weighing! \n Exception Message: {e}")
            raise e

    def calcUserProfiles(self) -> pd.DataFrame:
        """
        Thomas Mandelz
        calculates the userprofiles from ratings and featurematrix by matmul
        :return: pandas Dataframe with the userprofiles
        :rtype: pd.DataFrame
        """
        try:
            # Calculate Userprofiles by multiplication of ratingsMatrix.T with normalised, weighted Features
            # ((users x movies) x (movies x features)) = (users x features)
            return self.ratings.T @ self.numericFeatures

        except Exception as e:
            logging.error(
                f"Error while calculation Userprofiles \n Exception Message: {e}")
            raise e

    def calcPairwiseSimilarity(
        self, firstDF: pd.DataFrame, secondDF: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Thomas Mandelz
        calculates the pairwise distance of 2 dataframes, converts the distance to a similarity measure and returns it in a matrice
        :param pd.DataFrame firstDF: First Dataframe to measure similarity
        :param pd.DataFrame secondDF: Second Dataframe to measure similarity
        :return: pandas Dataframe with the similarity of the vector pairs from param dataframes
        :rtype: pd.DataFrame
        """
        try:
            if self.config.similarity == SimilarityEnum.COSINE:
                # Calculates the pairwise distance of userprofiles and movies, 0 = same vector, 1 = orthogonal, 2 = opposite vector
                # cosine distance = 1 - cosine similarity -> cosine similarity = 1 - cosine distance
                # with 1 - distance the similarity is calculated, -1 = opposite vector, 0 = orthogonal, 1 = same vector
                return pd.DataFrame(
                    1 - pairwise_distances(firstDF, secondDF,
                                           metric="cosine", n_jobs=-1)
                )
            elif self.config.similarity == SimilarityEnum.JACCARD:
                raise NotImplementedError("Jaccard is not used!")
            elif self.config.similarity == SimilarityEnum.PEARSON:
                # https://stackoverflow.com/questions/41823728/how-to-perform-correlation-between-two-dataframes-with-different-column-names
                n = len(firstDF)
                firstDF = firstDF.T
                secondDF = secondDF.T
                v1, v2 = firstDF.values, secondDF.values
                sums = np.multiply.outer(v2.sum(0), v1.sum(0))
                stds = np.multiply.outer(v2.std(0)+ 10**-7, v1.std(0)+ 10**-7)
                return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n, secondDF.columns, firstDF.columns).T
        except NotImplementedError as notex:
            logging.warning(
                f"Similarity Measure not implemented \n Exception Message: {notex}")
            raise e
        except Exception as e:
            logging.error(
                f"Error while calculating Pairwise Similarities for Dataframes \n Exception Message: {e}")
            raise e

    def predictSimilaritiesUser(self):
        """
        Thomas Mandelz
        Calculates Userprofiles from ratings and features, compares userprofiles to movies and returns nearest n Movies to the user
        :param pd.DataFrame ratingsMatrix: ratings matrix from the MatrixGenerator
        :return: returns a pandas Series with the the movie ID and the similarity measure asociated
        :rtype: pd.Series
        """
        # Calculate distances/similarites of Features vector space of movies to the userprofile vectors spaces
        similaritiesDf = self.calcPairwiseSimilarity(
            self.numericFeatures, self.userProfiles
        )
        # reassign indexes and columnames
        similaritiesDf.index = self.ratings.index
        similaritiesDf.columns = self.userProfiles.index

        return similaritiesDf

    def predictTopNForUser(
        self,
        users: list,
        n: int = 10,
        removeRatedMovies: bool = True,
    ) -> pd.Series:
        """
        Thomas Mandelz
        Calculates Userprofiles from ratings and features, compares userprofiles to movies and returns nearest n Movies to the user
        :param str userId: Userid of the user to make predictions
        :param int n: top n movies to return
        :param bool removeRatedMovies: removes already rated movies
        :return: returns a pandas Series with the the movie ID and the similarity measure asociated
        :rtype: pd.Series
        """
        # Ensure parameter is a list instance
        if isinstance(users, list) == False:
            raise Exception("users has to be a list.")
        # Top N for Combination of multiple Users
        if len(users) > 1:
            # get userprofiles and normalize them by dividing through the l2 norm, sum them to one profile
            combuserProfile = np.sum(self.userProfiles.loc[users].T / np.linalg.norm(self.userProfiles.loc[users], axis=1), axis=1)
            combuserProfile.columns = ["userCombination"]
            # Calculate distances/similarites of Features vector space of movies to the userprofile vectors spaces
            similaritiesDf = self.calcPairwiseSimilarity(
                self.numericFeatures, np.array([combuserProfile])
            )
            # reassign indexes and columnames
            similaritiesDf.index = self.ratings.index
            similaritiesDf.columns = combuserProfile.columns

            # get top n nearest movies for the user
            topNMovieIds = similaritiesDf["userCombination"].sort_values(
                ascending=False)

        # Top N for Single User
        elif len(users) == 1:
            similaritiesDf = self.predictSimilaritiesUser()
            # get top n nearest movies for the user
            topNMovieIds = similaritiesDf[users].sort_values(
                by=users, ascending=False)

        if removeRatedMovies:
            ratingsPerMovieCounts = np.count_nonzero(self.ratings.loc[:, users], axis=1)
            ratedMovieIds = self.ratings.index[ratingsPerMovieCounts > 0]
            topNMovieIds = topNMovieIds[~topNMovieIds.index.isin(ratedMovieIds)]

        return topNMovieIds.iloc[:n]

    def predictRatings(self):
        """
        at the moment not used
        """
        raise NotImplementedError()
        simDf = self.predictSimilaritiesUser()
        # Scale to (1 - 5) range
        predictedRatingDf = ((simDf + 1) * 2) + 1
        return predictedRatingDf

    def intersectionTopNMoviesUser(self, userList: list, n: int = 10, removeRatedMovies=True) -> pd.DataFrame:
        """
        Thomas Mandelz
        Gets Movie Recommendation Intersection of 2 or more users
        :param list userList: List with user Id's
        :param int n: top n movies to return
        :return: returns a pandas DataFrame with all similarities of the users, the mean similarity and the asociated movieId (index)
        :rtype: pd.DataFrame
        """
        # Ensure parameter is a list instance
        if isinstance(userList, list) == False:
            raise Exception("userList has to be a list.")
        # get similarities of all movies for each user
        movies = []
        for u in userList:
            movies.append(self.predictTopNForUser(
                [u], n=self.numericFeatures.shape[0], removeRatedMovies=removeRatedMovies))

        # Get Intersection of all users to the movies
        intersection = pd.concat(movies, axis=1, join='inner')
        # Calculate Mean similarity
        intersection["mean"] = np.mean(intersection, axis=1)
        # sort by mean similarity and return top n
        intersection = intersection.sort_values("mean", ascending=False)
        return intersection[:n]

    def combineUserProfiles(self, userList: list, combinedUserProfileName: str = "userpair") -> pd.Series:
        """
        Thomas Mandelz
        Combines Userprofiles and returns new similarities of a user Pair
        :param list userList: List with user Id's
        :param str combinedUserProfileName: New Name of the userPair, is set as column name
        :return: returns a pandas Series with all similarities of the new userpair to all movies
        :rtype: pd.Series
        """
        # Ensure parameter is a list instance
        if isinstance(userList, list) == False:
            raise Exception("userList has to be a list.")
        # get userprofiles by summing them together, divide by l2 norm for weighting
        combuserProfile = np.sum(self.userProfiles.loc[userList].T / np.linalg.norm(self.userProfiles.loc[userList], axis=1), axis=1)
        # Column name for the user Pair
        combuserProfile.columns = [combinedUserProfileName]
        # Calculate distances/similarites of Features vector space of movies to the userprofile vectors spaces
        similaritiesDf = self.calcPairwiseSimilarity(
            self.numericFeatures, combuserProfile
        )
        # reassign indexes and columnames
        similaritiesDf.index = self.ratings.index
        similaritiesDf.columns = combuserProfile.columns
        # sort similarity descending
        similaritiesDf = similaritiesDf[combinedUserProfileName].sort_values(
            ascending=False)
        return similaritiesDf

    def calcAppUserProfiles(self, movieIdUsers: list) -> None:
        """
        Jan Zwicky
        calculate user Profiles with movieId's.
        :param list movieIdUsers: List of lists with the movieId for multiple users
        """
        profiles = pd.DataFrame({})
        for n, movieId in enumerate(movieIdUsers):
            profiles = pd.concat((profiles, self.numericFeatures.loc[movieId,].sum().to_frame().transpose().set_index(np.array([str(n)]))))

        self.userProfiles = profiles


    def getMetadataCSV(self,movieId,movieName):
        movieId = list(movieId)
        movieName = list(movieName)
        metadataAll = pd.read_csv(r"./data/movies.csv",low_memory=False)
        indexes = metadataAll[metadataAll.original_title.isin(movieName)].movieId
        titles = metadataAll[metadataAll.movieId.isin(movieId)]["title.1"]
        descriptions = metadataAll[metadataAll.movieId.isin(movieId)].overview

        return indexes,titles,descriptions


    def getMovieIdbyName(self,movieName):
        indexes,titles,descs = self.getMetadataCSV("",movieName)
        return indexes
    def getMovieNamebyId(self,movieId):
        indexes,titles,descs = self.getMetadataCSV(movieId,"")
        return titles
    def getMovieDescbyId(self,movieId):
        indexes,titles,descs = self.getMetadataCSV(movieId,"")
        return descs