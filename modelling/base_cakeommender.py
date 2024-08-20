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
from libraries.weights import WeightGenerator
from cakeommender import Cakeommender
from evaluation import Evaluation


# %% [markdown]
"""
# Base Recommendersystem

In diesem Notebook wird das beste Recommendersystem gesucht ohne NLP-Ansätze.

Das Vorgehen wurde folgendermassen definiert:

* quantitativ die beste Kombination von Ähnlichkeitsmass und Ratingskala herausfinden.
* quantitativ mit der besten Kombination, die besten Feature-Normalisierung herausfinden.
* quantitativ die besten Gewichtungen der Daten herausfinden.
* qualitative Überprüfung des Recommenders

"""

# %% [markdown]
"""
## Kombination von Ähnlichkeitsmass und Ratingskala
"""
# %%
matrixBase = MatrixGenerator(metadata = True,genres = True,actors = False,directors=False)

# %%
resultBase = pd.DataFrame({})
for similarities in [SimilarityEnum.COSINE,SimilarityEnum.PEARSON]:
    for ratingScale in RatingScaleEnum:
        config = CakeConfig(
            {MatrixGenerator.CONST_KEY_METADATA: np.array(1),
            MatrixGenerator.CONST_KEY_GENRES: np.array(1),
            MatrixGenerator.CONST_KEY_ACTORS: np.array(1),
            MatrixGenerator.CONST_KEY_DIRECTORS: np.array(1)},
            similarities,
            ratingScale,
            FeatureNormalizationEnum.ZSCORE
            )
        eval = Evaluation("BaseCake",config,matrixBase)
        precisionMean,precisionStd = eval.precision()
        novelty = eval.novelty()
        topNPrecision = eval.topNPrecision()
        resultBase = pd.concat((resultBase,pd.DataFrame({"RatingScale":[ratingScale.name],"Similarity":[similarities.name],"precisionMean":[precisionMean],"precisionStd":[precisionStd], "novelty": [novelty],"topNPrecision":[topNPrecision]})))

# %%
fig = plt.figure(figsize=(20,10),dpi = 80)
gs = fig.add_gridspec(4, 2, hspace= 0.4)


# subplot Precision
x = resultBase["RatingScale"] + " " + resultBase["Similarity"]
y = resultBase["precisionMean"]
y_error = resultBase["precisionStd"]

fig.suptitle("Quantitative Evaluierung von Ähnlichkeitsmassen und Rating Skalen", fontsize = 20)
precision = fig.add_subplot(gs[:2, :2])
sns.scatterplot(x=x, y=y)
precision.errorbar(x, y,
             yerr = y_error,
             fmt ='o')
precision.set_title("Precision")
precision.set_xlabel("RatingScale - Similarity")
precision.set_ylabel("Precision")

# sublot TopN-Precision
topNPrecision = fig.add_subplot(gs[2:4, 0 ])
sns.scatterplot(x='RatingScale', y='topNPrecision', ax=topNPrecision, data=resultBase, hue='Similarity')
topNPrecision.set_title("Top-10 Precision")
topNPrecision.set_ylabel("Precision")

# subplot Novelty
novelty = fig.add_subplot(gs[2:4, 1])
novelty.set_title("Top-10 Novelty")
sns.scatterplot(x='RatingScale', y='novelty', ax=novelty, data=resultBase, hue='Similarity')
plt.show()

# %% [markdown]
"""
**Ratingskala:** Die tertiäre Skala (-1, 0, 1) hat die beste Precision und Novelty. Sie hat aber die schlechteste Top-10 Precision.
Da bei der Top-10 Precision der Unterschied deutlich kleiner ist, werden wir für die weiteren Berechnungen die tertiäre Skala verwenden. Es ist erstaunlich, dass sie besser abschneidet als die detailliertere Standardisierung. Anscheinend ist diese genaue Einteilung nicht hilfreich und eine einfache Separation von gut, schlecht und nicht bewertet ist absolut ausreichend.
<br><br>

**Ähnlichkeitsmass:** Die beiden Ähnlichkeitsmasse erzielen fast die gleichen Resultate. Für die weiteren Berechnungen entscheiden wir uns für die Kosinusähnlichkeit, da sie zu leicht besseren Ergebnissen führt.

## Beste Kombination mit optimaler Feature-Normalisierung
"""
# %%
resultFeatures = pd.DataFrame({})
for featureNormalisation in FeatureNormalizationEnum:
    config = CakeConfig(
            {MatrixGenerator.CONST_KEY_METADATA: np.array(1),
            MatrixGenerator.CONST_KEY_GENRES: np.array(1),
            MatrixGenerator.CONST_KEY_ACTORS: np.array(1),
            MatrixGenerator.CONST_KEY_DIRECTORS: np.array(1)},
            SimilarityEnum.COSINE,
            RatingScaleEnum.TERTIARY,
            featureNormalisation
            )
    eval = Evaluation("BaseCake",config,matrixBase)
    precisionMean,precisionStd = eval.precision()
    novelty = eval.novelty()
    topNPrecision = eval.topNPrecision()
    resultFeatures= pd.concat((resultFeatures,pd.DataFrame({"FeatureNormalization":[featureNormalisation.name],"precisionMean":[precisionMean],"precisionStd":[precisionStd], "novelty": [novelty],"topNPrecision":[topNPrecision]})))

# %%
# subplot Precision
x = resultFeatures["FeatureNormalization"]
y = resultFeatures["precisionMean"]
y_error = resultFeatures["precisionStd"]

fig2 = plt.figure(figsize=(20,10),dpi = 80)
gs2 = fig2.add_gridspec(2, 3, hspace= 0.4)

fig2.suptitle("Quantitative Evaluierung von Feature Normalisierungen ",fontsize = 20)
precision = fig2.add_subplot(gs2[:1, 0])
sns.scatterplot(x=x, y=y)
precision.errorbar(x, y,
             yerr = y_error,
             fmt ='o')
precision.set_title("Precision")
precision.set_xlabel("FeatureNormalization")
precision.set_ylabel("Precision")

# sublot TopN-Precision
topNPrecision = fig2.add_subplot(gs2[:1, 1])
sns.scatterplot(x = 'FeatureNormalization', y = 'topNPrecision',ax =topNPrecision, data=resultFeatures)
topNPrecision.set_title("Top-10 Precision")
topNPrecision.set_ylabel("Precision")

# subplot Novelty
novelty = fig2.add_subplot(gs2[:1,2])
novelty.set_title("Top-10 Novelty")
sns.scatterplot(x = 'FeatureNormalization', y = 'novelty',ax = novelty, data=resultFeatures)
plt.show()
# %% [markdown]
"""
**Feature Normalisierung:**

Die ZScore Normalisierung erzielt die besten Resultate bei allen Metriken. Aus diesem Grund werden wir die ZScore Normalisierung für die weiteren Recommendersysteme nutzen.
"""
# %% [markdown]
"""
## Beste Gewichtungen
### Quantitative Untersuchung
"""
# %%
matrixBase = MatrixGenerator(metadata=True, genres=True, actors=True, directors=True)
resultFeaturesWeights = pd.DataFrame({})
weights = [[1,1,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0],[1,1,0.4,0]]
for weight in weights:
    config = CakeConfig(
            {MatrixGenerator.CONST_KEY_METADATA: np.array(weight[0]),
            MatrixGenerator.CONST_KEY_GENRES: np.array(weight[1]),
            MatrixGenerator.CONST_KEY_ACTORS: np.array(weight[2]),
            MatrixGenerator.CONST_KEY_DIRECTORS: np.array(weight[3])},
            SimilarityEnum.COSINE,
            RatingScaleEnum.TERTIARY,
            FeatureNormalizationEnum.ZSCORE
            )
    eval = Evaluation("BaseCake",config,matrixBase)
    precisionMean,precisionStd = eval.precision()
    novelty = eval.novelty()
    topNPrecision = eval.topNPrecision()
    #weight = "Metadaten: " + str(weight[0]*100) + "%\n" + "Genres: " + str(weight[1]*100) + "%\n" + "Actors: " +str(weight[2]*100) + "%\n" + "Directors: " + str(weight[3]*100) + "%"
    weight = f"({str(weight[0])},{str(weight[1])},{str(weight[2])},{str(weight[3])})"
    resultFeaturesWeights= pd.concat((resultFeaturesWeights,pd.DataFrame({"featureWeights":weight,"precisionMean":[precisionMean],"precisionStd":[precisionStd], "novelty": [novelty],"topNPrecision":[topNPrecision]})))
# %%
x = resultFeaturesWeights["featureWeights"]
y = resultFeaturesWeights["precisionMean"]
y_error = resultFeaturesWeights["precisionStd"]

fig2 = plt.figure(figsize=(20,10),dpi = 80)
gs2 = fig2.add_gridspec(2, 3, hspace= 0.4)

fig2.suptitle("Quantitative Evaluierung von Gewichtungen",fontsize = 18)
precision = fig2.add_subplot(gs2[:1, 0])
sns.scatterplot(x=x, y=y)
precision.errorbar(x, y,
             yerr = y_error,
             fmt ='o')
precision.set_title("Precision")
precision.set_xlabel("featureWeights")
precision.set_ylabel("Precision")

# sublot TopN-Precision
topNPrecision = fig2.add_subplot(gs2[:1, 1])
topNPrecision.set_xlabel('Gewichtungen Features', fontsize=14)
sns.scatterplot(x = 'featureWeights', y = 'topNPrecision',ax =topNPrecision, data=resultFeaturesWeights)
topNPrecision.set_title("Top-10 Precision")
topNPrecision.set_ylabel("Precision")

# subplot Novelty
novelty = fig2.add_subplot(gs2[:1,2])
novelty.set_title("Top-10 Novelty")
novelty.set_xlabel("")
sns.scatterplot(x = 'featureWeights', y = 'novelty',ax = novelty, data=resultFeaturesWeights)
plt.subplots_adjust(left=0,
                    bottom=0.1,
                    right=1,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.4)

plt.show()
# %% [markdown]
"""
*Lesebeispiel: Auf der X-Achse sind die Gewichtungen der Metadaten, Genres, Actors und Directors angegeben. So sind jeweils im zweiten Modell keine Metadaten vorhanden, 100 % Gewichtung der Genres, Actors und Directors.*

Die beste Precision (sowohl gesamthaft als auch über die Top-10) kann mit folgender Gewichtung erreicht werden:

* 100 % Metadaten
* 100 % Genres
* 40 % Actors
* 0 % Directors

### Qualitative Untersuchung
**Metadaten:**
"""
# %%
matrixBase = MatrixGenerator(metadata=True)
config = CakeConfig(
            {
            MatrixGenerator.CONST_KEY_METADATA: np.array(1),
            #MatrixGenerator.CONST_KEY_GENRES: np.array(1),
            #MatrixGenerator.CONST_KEY_ACTORS: np.array(1),
            #MatrixGenerator.CONST_KEY_DIRECTORS: np.array(1)
            },
            SimilarityEnum.COSINE,
            RatingScaleEnum.TERTIARY,
            FeatureNormalizationEnum.ZSCORE
            )
eval = Evaluation("BaseCake",config,matrixBase)

eval.iterCleveland()
# %% [markdown]
"""
*Lesebeispiel der letzten Grafik: Die Werte auf der x-Achse entsprechen der Präferenz der einzelnen User für die jeweiligen Feature. Nutzer 2 hat eine Abneigung gegenüber hohen Filmen mit hohen Umsätzen (`revenue`, er mag wohl keine Blockbuster) und Nutzer 1 mag dafür keine langen Filme. In dieser Grafik sind alle Werte negativ, das heisst, dass beide User keine Vorlieben gegenüber langen Filmen oder sehr erfolgreichen Filmen haben, die auch viel Budget benötigten. Bei positiven Werten wäre das Gegenteil der Fall und die Nutzer würden hohe Werte bei diesen Features bevorzugen. Der gelbe Punkt zeigt, wie die vorgeschlagenen Filme den Wertebereich des Features abdecken. Hohe Werte bedeuten hier, dass diese Filme auch hohe Werte beim jeweiligen Feature haben. Idealerweise liegt dieser gelbe Punkt zwischen den Punkten für die beiden User.*
"""
# %% [markdown]
"""
Qualitativ funktioniert das Recommendersystem mit den Metadaten gut.
Die meisten Werte der kombinierten Filmprofile befinden sich zwischen den beiden User.
Dies sagt, aber noch nicht viel darüber aus, wie gut das Recommendersystem tatsächlich ist.
Da es für einen Recommender, welcher genügend Filme zur Auswahl hat, simpel ist, ähnliche Filme vorherzusagen.

Um es mit zusätzlichen Feature zu testen, wird der Cleveland Plot mit den Genres gemacht.
"""
# %% [markdown]
"""
**Genres:**
"""
# %%
matrixBase = MatrixGenerator(genres=True)
config = CakeConfig(
            {
            #MatrixGenerator.CONST_KEY_METADATA: np.array(1),
            MatrixGenerator.CONST_KEY_GENRES: np.array(1),
            #MatrixGenerator.CONST_KEY_ACTORS: np.array(1),
            #MatrixGenerator.CONST_KEY_DIRECTORS: np.array(1)
            },
            SimilarityEnum.COSINE,
            RatingScaleEnum.TERTIARY,
            FeatureNormalizationEnum.ZSCORE
            )
eval = Evaluation("BaseCake",config,matrixBase)

eval.iterCleveland()
# %% [markdown]
"""
Das kombinierte Filmprofil, befindet sich nicht immer zwischen den beiden Userprofilen. Entscheidend ist, dass die Tendenz, ob ein Genres eines Users geschätzt wird oder nicht, wichtig. Wie stark diese Ausprägung ist, ist zweitrangig. Die Visualisierung zeigt bei fast allen Usern ziemlich gute Resultate für das kombinierte Userprofil.
"""
# %%
%reset -f