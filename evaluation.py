# %% [markdown]
## Evaluation
"""Dieses Notebook wurde verwendet für die Funktionen der Evaluierung"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import platform

from cakeommender import Cakeommender
from libraries.pipeconfig import (
    CakeConfig,
    RatingScaleEnum,
)
from libraries.matrix import MatrixGenerator, MatrixLoader

isPDFGenerator = platform.system() == "Linux"

# %%
class Evaluation:
    CONST_RATING_FILE_PATHS = {
        RatingScaleEnum.BINARY: "./data/userItem_bin_long.csv",
        RatingScaleEnum.TERTIARY: "./data/userItem_tert_long.csv",
        RatingScaleEnum.STANDARDISED: "./data/userItem_stand_long.csv"
    }

    cached_popularities = None
    cached_unknown_ratings = None

    def __init__(
        self,
        name: str,
        config: CakeConfig,
        matrixGenerator: MatrixGenerator
    ) -> None:

        """
        Jan
        init method for class to generate config object, an instance of a cakeommender, a similiraty matrix and a cut.

        :param str name: Name of the Cakeommender instance
        :param BaseCakeConfig config: cakeommender configuration object
        :param MatrixGenerator matrixGenerator: MatrixGenerator instance to generate feature matrix
        """
        self.config = config
        self.matrixGenerator = matrixGenerator
        self.name = name
        self.baseModel = Cakeommender(self.name, self.config, self.matrixGenerator, verbose=True)
        self.baseSimilarity = self.baseModel.predictSimilaritiesUser()
        self.cut = np.quantile(self.baseSimilarity.values, 0.6)

        # load ratings in long format
        if config.ratingScale not in self.CONST_RATING_FILE_PATHS:
            raise NotImplementedError()

        filename = self.CONST_RATING_FILE_PATHS[config.ratingScale]
        self.matrixLoader = MatrixLoader()
        self.ratingsLong = self.matrixLoader.loadLong(filename)

        # load the good rated movies
        self.goodRated = self.matrixLoader.loadLong(
            self.CONST_RATING_FILE_PATHS[RatingScaleEnum.BINARY]
        ).rename(columns={"rating": "goodMovie"})

    def precision(self, seed : int = 54)-> list:
        """
        Jan
        Calculate the Precision for 5 Cross-Validation folds.

        :param int seed: set the seed of the kfold split.
        :return : Precisions of 5 folds
        :rtype : list
        """
        # Ratings for more than 5 user available
        longFormat = self.ratingsLong.groupby(['userId']).filter(lambda x: len(x) >= 5)
        longFormat = pd.merge(
            longFormat,
            self.goodRated,
            on = ["userId", "movieId"],
            how = "left"
        )
        longFormat['rated'] = 1

        k = 5
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        precisions = []

        it = tqdm(skf.split(longFormat["movieId"], longFormat["userId"]), total=k, leave=not isPDFGenerator)
        for trainIndex, testIndex in it:
            # Train test split
            testData = longFormat.iloc[testIndex]
            trainData = longFormat.iloc[trainIndex]

            # Train split as sparse matrix
            ratings = pd.pivot(trainData, index="movieId", columns="userId", values="rating")
            modelFold = Cakeommender(self.name, self.config, self.matrixGenerator, ratingsSplit=ratings, verbose=True, progress=False)

            # calc similarities
            similarities = modelFold.predictSimilaritiesUser()
            positives = similarities >= self.cut

            ground_truth = pd.pivot(testData, index="movieId", columns="userId", values="goodMovie")
            rated = pd.pivot(testData, index="movieId", columns="userId", values="rated")

            tp = np.sum(positives * ground_truth).sum()
            tpfp = np.sum(positives * rated).sum()
            precisions.append(tp / tpfp)

            it.set_postfix({'mean': np.mean(precisions).round(3), 'std': np.std(precisions).round(3)})

        return np.mean(precisions), np.std(precisions)

    def topNPrecision(self, lengthTopN : int = 10, seed : int = 62) -> float:
        """
        Jan
        Calculate the Precision of TopN-list of the users.
        :param int lengthTopN: length of TopN-List
        :param int seed: set the seed of the sample.
        :return: Precisions
        :rtype: float
        """

        # select user which rated 3 + 2 * lenght of TopN
        longFormat = self.ratingsLong.groupby(['userId']).filter(lambda x: len(x) >= 3 + 2 * lengthTopN)
        longFormat = pd.merge(longFormat, self.goodRated.loc[:,["userId", "movieId", "goodMovie"]], on=["userId", "movieId"], how="left")

        # train test split
        sampleOfUsers = longFormat.groupby("userId").sample(n=3, random_state=seed)
        testData = pd.merge(longFormat,sampleOfUsers.loc[:,['movieId', 'userId']], on=['userId', 'movieId'], how="outer", indicator=True)
        testData = testData[testData['_merge'] == 'left_only']

        # select user with lengthTopN good Rating and lengthTopN bad Ratings
        groupedUser = testData.groupby(["userId","goodMovie"]).size() > lengthTopN
        groupedUser = pd.DataFrame(groupedUser).reset_index()
        usersWithEnoughRatings = groupedUser[groupedUser[0]]["userId"]
        usersWithEnoughRatings = usersWithEnoughRatings[usersWithEnoughRatings.duplicated()]
        testData = testData[testData["userId"].isin(usersWithEnoughRatings)]
        sampleOfUsers = sampleOfUsers[sampleOfUsers["userId"].isin(usersWithEnoughRatings)]

        # model with less ratings
        rating = pd.pivot(sampleOfUsers, index="movieId", columns="userId", values="rating")
        model = Cakeommender(self.name, self.config,self.matrixGenerator, ratingsSplit=rating, verbose=True)

        # predict model
        predictedSimilarities = model.predictSimilaritiesUser()
        predictedRatingsLong = pd.melt(predictedSimilarities.reset_index(names='movieId'), id_vars=["movieId"], value_vars=predictedSimilarities.columns).rename(columns={"variable": "userId"})
        #predictedRatingsLong["goodRated"] = predictedRatingsLong["value"] > self.cut

        # calc precision
        trainTestData = pd.merge(testData, predictedRatingsLong, how="left", on=["movieId", "userId"])
        trainTestData = trainTestData.groupby("userId").apply(lambda grp: grp.nlargest(lengthTopN, "value"))
        return sum(trainTestData["goodMovie"]) / len(trainTestData)

    def novelty(self, lengthTopN: int = 10) -> float:
        """
        Jan
        Quantitative evaluation metric novelty
        https://spaces.technik.fhnw.ch/spaces/recommender-systems/beitraege/recommender-system-evaluierung-coverage-und-novelty
        :param int lengthTopN: length of TopN-List
        :return: novelty value
        :rtype: float
        """
        similarities = self.baseSimilarity * Evaluation.get_unknown_ratings()
        n_movies = similarities.shape[0]

        thresholds = similarities.quantile(1 - lengthTopN / n_movies)
        movies_count = (similarities >= thresholds).sum(axis=1).rename('counts').to_frame()
        movies_count['popularity'] = Evaluation.calculate_popularities() * movies_count.counts
        movies_count['popularity'] = movies_count['popularity'].fillna(0)
        return -np.sum(movies_count['popularity']) / movies_count.counts.sum()

    def cleveland(self, user1 : str, user1Description : str, user2 : str, user2Description: str, ax = None) -> None:
        """
        Jan
        Qualitative evaluation with a cleveland plot and certain user

        :param str user1 : first user id
        :param str user1Description : first user description
        :param str user2 : second user id
        :param str user2Description : second user description
        """
        # https://www.python-graph-gallery.com/184-lollipop-plot-with-2-groups
        axNone = False
        if ax is None:
            _ , ax = plt.subplots()
            axNone = True

        # calc combined profil
        topN = self.baseModel.predictTopNForUser([user1,user2],10,False)
        topNFeatures = self.baseModel.numericFeatures.loc[topN.index].sum(axis = 0)

        userProfiles = self.baseModel.userProfiles.loc[[user1,user2]]

        # Reorder it following the values of the first value:
        profiles = pd.concat([userProfiles,topNFeatures.to_frame().T])
        #profiles = profiles/ np.linalg.norm(profiles, axis=0)
        profiles = profiles.transpose().reset_index()
        profiles[[user1,user2,0]] = profiles[[user1,user2,0]]/np.linalg.norm(profiles[[user1,user2,0]], axis=0)
        ProfilesOrdered = profiles.sort_values(by='level_1')
        Range = range(1, len(profiles.index)+1)

        # The horizontal plot is made using the hline function
        ax.hlines(y=Range, xmin=ProfilesOrdered[user1], xmax=ProfilesOrdered[user2], color='grey', alpha=0.4)
        ax.scatter(ProfilesOrdered[user1], Range, color='green', alpha=1, label='Profil User 1')
        ax.scatter(ProfilesOrdered[user2], Range, color='red', alpha=1, label='Profil User 2')
        ax.scatter(ProfilesOrdered[0], Range, color='orange', alpha=1, label='Filmvorschläge')
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        # Add title and axis names
        ax.set_yticks(Range, ProfilesOrdered['level_1'])
        ax.set_title("Nutzer 1: " + user1Description + "\n Nutzer 2: "+ user2Description)
        #plt.c("user 1 :" + user1Description + "\n user 2 : "+ user2Description)
        ax.set_xlabel('Präferenz')
        ax.set_ylabel('Features')

        if axNone:
            plt.show()
        else:
            return ax

    def iterCleveland(self) -> None:
        """
        Jan
        Qualitative evaluation with a cleveland plot and certain users for the qualitative Evaluation
        """
        qualitativeUsers = pd.read_csv("data/userQualitativeEvaluation.csv", dtype=str)

        fig, axs = plt.subplots(5, 2, figsize = (20, 30))
        axs = axs.flatten()

        for n, x in enumerate(range(10)):
            self.cleveland(*tuple(qualitativeUsers.iloc[x].astype(str)), axs[n])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def calculate_popularities():
        if Evaluation.cached_popularities is not None:
            return Evaluation.cached_popularities

        ratings = MatrixLoader().loadWide('./movielens_data/ratings.csv').reset_index(names='movieId')
        moviesPercentage = ratings.groupby('movieId')['rating'].count()
        Evaluation.cached_popularities = np.log2(moviesPercentage / ratings['movieId'].nunique()).rename('popularity')
        return Evaluation.cached_popularities

    @staticmethod
    def get_unknown_ratings():
        if Evaluation.cached_unknown_ratings is not None:
            return Evaluation.cached_unknown_ratings

        unknown_ratings = MatrixLoader().loadWide('./data/userItem_stand.csv') # doesn't matter which of the sparse files
        unknown_ratings = unknown_ratings.isna().astype('int')
        Evaluation.cached_unknown_ratings = unknown_ratings
        return unknown_ratings

def plot_results(
    x: list[str],
    precisions: list[float],
    precision_errs: list[float],
    topNPrecisions: list[float],
    novelties: list[float],
    title: str,
    topNLength: int = 10
) -> plt.figure:
    """
    Joseph Weibel
    Creates three scatter plots showing precision (including error bars), Top-N precision and Top-N novelty.

    :param x list[str]: names of the models
    :param precisions list[float]: mean precision values of the models in the same order as in x
    :param precision_errs list[float]: std precision values of the models in the same order as in x
    :param topNPrecisions list[float]: Top-N precision values of the models in the same order as in x
    :param novelties list[float]: Top-N novelty values of the models in the same order as in x
    :param title str: title describing what kind of models are compared in the plots
    :param topNLength int: the number of movies in the recommendations used for topNPrecisions and novelties
    :return: plot with three subplots
    :rtype: plt.figure
    """

    fig = plt.figure(figsize=(30, 10), dpi=100)
    gs = fig.add_gridspec(2, 3, hspace=0.4)

    fig.suptitle(f'Quantitative Evaluierung von {title}', fontsize=20)
    precision_plot = fig.add_subplot(gs[:1, 0])
    precision_plot.scatter(x, precisions)
    precision_plot.errorbar(x, precisions, yerr=precision_errs, fmt='o')
    precision_plot.set_title('Precision')
    precision_plot.set_xlabel('')
    precision_plot.set_ylabel('Precision')

    # sublot TopN-Precision
    topNPrecision_plot = fig.add_subplot(gs[:1, 1])
    sns.scatterplot(x=x, y=topNPrecisions, ax=topNPrecision_plot)
    topNPrecision_plot.set_title(f'Top-{topNLength} Precision')
    topNPrecision_plot.ticklabel_format(axis='y', useOffset=False)
    topNPrecision_plot.set_xlabel('')
    topNPrecision_plot.set_ylabel('Precision')

    # subplot Novelty
    novelty_plot = fig.add_subplot(gs[:1,2])
    sns.scatterplot(x=x, y=novelties, ax=novelty_plot)
    novelty_plot.set_title(f'Top-{topNLength} Novelty')
    novelty_plot.ticklabel_format(axis='y', useOffset=False)
    novelty_plot.set_xlabel('')
    novelty_plot.set_ylabel('Novelty')

    return fig
