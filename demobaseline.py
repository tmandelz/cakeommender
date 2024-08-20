# %%
# add imports
from cakeommender import Cakeommender
from libraries.pipeconfig import (
    CakeConfig,
    FeatureNormalizationEnum,
    SimilarityEnum,
    RatingScaleEnum,
)
from libraries.matrix import MatrixGenerator
import numpy as np

# %%
# Create Featurematrix for Baserecommender
matrixBase = MatrixGenerator(
    metadata=True, genres=True, actors=True, directors=True)
# Create Config for Baserecommender
configBase = CakeConfig(
    {
        MatrixGenerator.CONST_KEY_METADATA: np.array(1),
        MatrixGenerator.CONST_KEY_GENRES: np.array(1),
        MatrixGenerator.CONST_KEY_ACTORS: np.array(1),
        MatrixGenerator.CONST_KEY_DIRECTORS: np.array(1),
    },
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE
)
# Init Baserecommender
baseModel = Cakeommender("baseModel", configBase, matrixBase)

# %%
# Predict Top N Movies for User 190304 and with Baserecommender
predictions = baseModel.predictTopNForUser(
    users=["190304"], removeRatedMovies=False)
print(f"BaseRecommender:\nUser 190304 predictions (movieID) and similarity values:\n")
print(predictions)
# %%
# Create Featurematrix for SBERTBaserecommender
matrixBaseSbert = MatrixGenerator(
    metadata=True, genres=True, actors=True, directors=True, sbertEmbeddings='data/movies_sbert_5d.csv')
# Create Config for SBERTBaserecommender
config = CakeConfig(
    {
        MatrixGenerator.CONST_KEY_METADATA: np.array(1),
        MatrixGenerator.CONST_KEY_GENRES: np.array(1),
        MatrixGenerator.CONST_KEY_ACTORS: np.array(0.4),
        MatrixGenerator.CONST_KEY_DIRECTORS: np.array(0),
        MatrixGenerator.CONST_KEY_SBERT: np.array(0.6)
    },
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE
)
# Init SBERTBaserecommender
bestModel = Cakeommender("bestModel", config, matrixBaseSbert)
# %%
# Predict Top N Movies for User 190304 and with SBERTBaserecommender
predictions = bestModel.predictTopNForUser(
    users=["190304"], removeRatedMovies=False)
print(f"SBERTBaserecommender:\nUser 190304 predictions (movieID) and similarity values:\n")
print(predictions)


# %%
