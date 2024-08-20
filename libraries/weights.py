import numpy as np
import pandas as pd
import logging


class WeightGenerator(object):
    """
    Thomas Mandelz
    Class to Generate Feature weights
    """

    def __init__(self) -> None:
        """
        Thomas Mandelz
        init method for class to generate feature weights
        """
        pass

    def createWeights(self, weightsDict: dict, featureMatrix: pd.DataFrame, nonNumericColumns: list) -> np.array:
        """
        Thomas Mandelz
        Function which creates a numpy array of the length of columns in the featurematrix with a weight attributed to each column group eg. "actors"
        :param dict weightsDict: dictionary which includes the column group (eg. "actor") and its featureweight as a float Example:{'actor' : np.array(0.5), 'crew' : np.array(0.2)}
        :param pd.DataFrame featureMatrix: Featurematrix to generate the weights for
        :return: 1 x m numpy array with featureweights
        :rtype: np.array
        """
        try:
            indexes = featureMatrix.columns.get_level_values(1).isin(nonNumericColumns)
            numeric_features = featureMatrix.loc[:, ~indexes]
            weights = np.array(
                [weightsDict[key] for key in numeric_features.columns.get_level_values(0)])
        except Exception as e:
            logging.error(
                f"Error while creating weights array! \n Exception Message: {e}")
            raise e
        return weights

    def createWeightsGranular(self):
        raise NotImplementedError()
