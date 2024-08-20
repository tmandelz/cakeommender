from enum import Enum
import numpy as np


# Classes
class SimilarityEnum(Enum):
    """
    Thomas Mandelz
    Enum Class to represent Similarity Measures
    """
    JACCARD = 1
    COSINE = 2
    PEARSON = 3
    DEEP = 4


class FeatureNormalizationEnum(Enum):
    """
    Thomas Mandelz
    Enum Class to represent Feature Normalisation Measures
    """
    ZSCORE = 1
    MINMAX = 2
    NONE = 3


class RatingScaleEnum(Enum):
    """
    Thomas Mandelz
    Enum Class to represent Rating Scales
    """
    BINARY = 1
    TERTIARY = 2
    STANDARDISED = 3


# Configuration Objects
class CakeConfig(object):
    """
    Thomas Mandelz
    Base Class which represents a config for any cakeommender
    """

    def __init__(self, weightsDistribution: dict, similarity: SimilarityEnum, ratingScale: RatingScaleEnum, featureNormalisation: FeatureNormalizationEnum) -> None:
        """
        Thomas Mandelz
        init method for class to generate config object
        :param dict weightsDistribution: weights dictionary for the categories and their weight
        :param SimilarityEnum similarity: similarity measure
        :param RatingScaleEnum ratingScale: rating scale
        :param FeatureNormalizationEnum featureNormalisation: featureNormalisation scale
        """
        # attributes
        self.weightsDistribution = weightsDistribution
        self.similarity = similarity
        self.ratingScale = ratingScale
        self.featureNormalisation = featureNormalisation

    def __str__(self) -> str:
        """
        Thomas Mandelz
        Overriden to string method for logging
        """
        return f"{self.__class__} with:\n Weights array: {self.weightsDistribution} \n Similarity: {self.similarity} \n RatingScale: {self.ratingScale} \n FeatureNormalisation: {self.featureNormalisation} \n"
