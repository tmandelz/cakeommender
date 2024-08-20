# %%
import os
if os.getcwd().endswith('modelling'):
    os.chdir('..')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from libraries.pipeconfig import (
    CakeConfig,
    FeatureNormalizationEnum,
    SimilarityEnum,
    RatingScaleEnum,
)
from libraries.matrix import MatrixGenerator
from libraries.feature_evaluation import plotSimiliarities
from cakeommender import Cakeommender
from evaluation import Evaluation

# %% [markdown]
"""
# Recommendersystem mit zufälligen Featurematrizen

Dieses erste Recommendersystem erstellen wir mit zufälligen Werten in der Featurematrix. Dieses können wir verwenden, um die weiteren Recommendersysteme an diesem messen zu können. Systeme, die schlechtere Werte als dieses zufällige System liefern, sind unbrauchbar.

Wir befüllen nun eine Matrix mit zufälligen normalverteilten Werten zwischen 0 und 1, so dass für jeden Film drei zufällige Features vorhanden sind.
"""

# %%
meta = pd.read_csv('data/movies_meta.csv', index_col='movieId')
features = pd.DataFrame(np.random.normal(0, 1, (meta.shape[0], 3)), index=meta.index)

# %% [markdown]
"""
Die Ähnlichkeiten sind nun gleichmässig zwischen -1 und 1 verteilt.
"""

# %%
fig = plotSimiliarities(features, 'random')

# %%
features_file = 'data/movies_random.csv'
features.to_csv(features_file)

# %% [markdown]
"""
## Modellierung des Recommendersystems

Mit diesen zufälligen Werten erstellen wir nun ein Recommendersystem.
"""

# %%
matrix = MatrixGenerator(random=True)

# %%
config = CakeConfig(
    {MatrixGenerator.CONST_KEY_RANDOM: np.array(1)},
    SimilarityEnum.COSINE,
    RatingScaleEnum.STANDARDISED,
    FeatureNormalizationEnum.ZSCORE
)

eval = Evaluation('Random', config, matrix)

# %%
mean_precision, std_precision = eval.precision()
topn_precision = eval.topNPrecision()
novelty = eval.novelty()

# %% [markdown]
"""
Die Precision liegt bei ca. 55 %, was etwa dem Verhältnis an guten Ratings im Testdatenset entspricht.
"""

# %%
plt.figure(figsize = (10, 5))
plt.errorbar(['random'], mean_precision, yerr = std_precision, fmt ='o')
plt.title('Precision Random-Recommendersystem')
plt.ylabel('Precision')
plt.show()

# %%
print('Precision Top-10-Liste:', topn_precision)
print('Novelty Top-10-Liste:', novelty)

# %% [markdown]
"""
Für alle weiteren Recommender-Systeme sollten also bessere Precision-Werte erzielt werden können.
"""
# %%
%reset -f