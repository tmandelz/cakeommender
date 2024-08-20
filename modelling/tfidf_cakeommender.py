# %%
import os
if os.getcwd().endswith('modelling'):
    os.chdir('..')

# %%
import platform
os = platform.system()

if os == "Linux":
    isNotpdfGenerator = False
else:
    isNotpdfGenerator = True
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from libraries.tfidf import tokenize
from libraries.pipeconfig import (
    CakeConfig,
    FeatureNormalizationEnum,
    SimilarityEnum,
    RatingScaleEnum,
)
from libraries.matrix import MatrixGenerator
from libraries.weights import WeightGenerator
from cakeommender import Cakeommender
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from evaluation import Evaluation

# %% [markdown]
"""
# Recommendersystem mit TFIDF-Tokens

Für unser erstes Recommendersystem auf Basis der Filmbeschreibungen verwenden wir *TFIDF* um diese in Tokens umzuwandeln. Diese Tokens verwenden wir anschliessend als Featurematrix für das Recommendersystem.

## Steckbrief

Bei TFIDF wird die Häufigkeit eines Wortes in einem Text bestimmt und diese in Relation zur Häufigkeit desselben Wortes in allen Texten gesetzt. So werden seltene Wörter stärker gewichtet als solche, die in jedem Text vorkommen. Die resultierende Featurematrix enthält dann pro Wort ein Attribut, dass dessen TFIDF-Score für den jeweiligen Text bestimmt.

Der Score wird über folgende Formel bestimmt, wobei $tf(x)$ die Anzahl Vorkomnisse des Wortes $x$ im Text $t$ definiert und $df(x)$ die Anzahl Texte angibt, in welchen das Wort $x$ vorkommt. $n$ definiert die gesamte Anzahl Texte.

$$tfidf(x, t) = tf(x, t) \cdot log(\frac{n}{df(x)})$$

Das Resultat beträgt 0, wenn das Wort nicht im Text vorkommt, oder wenn es in allen Texten vorkommt. Aufgrund des Logarithmus werden seltene Worte viel stärker gewichtet und so können auch ziemlich hohe TFIDF-Scores erzielt werden. Es gibt entsprechend keinen möglichen Maximalwert für diese Formel.

### Pre-Processing

Die Anzahl Features durch die TFIDF-Transformation ergibt sich grundsätzlich aus der Anzahl unterschiedlicher Worte in der gesamten Textsammlung. Dies kann zu einer riesigen Featurematrix führen, was eine weitere Verabeitung stark verlangsamen kann. Insbesondere wenn die Textsammlung viele Worte beinhaltet, die nur sehr selten vorkommen, kann dies die Matrix je nach Anwendungsfall unnötig aufblähen.

Um die Matrix klein zu halten, bietet es sich an gewisse Pre-Processing-Schritte vorzunehmen. Jedoch eignen sich nicht alle Möglichkeiten für alle Sprachen und Anwendungsfälle.

* **lowercase**: Durch das Kleinschreiben der Worte entstehen keine Token-Duplikate, wenn ein Wort innerhalb eines Satzes vorkommt und einmal am Satzanfang, wo der erste Buchstabe grossgeschrieben wird. In Englisch funktioniert dies gut, aber in Sprachen wie Deutsch, in welcher zum Beispiel Nomen grossgeschrieben werden, kann dies dazu führen, dass Wörter mit einer unterschiedlichen Bedeutung zum selben Token zusammengefasst werden.

* **lemmatisation**: Wenn alle Wörter zuerst in ihre Grundform gebracht werden, kann die Anzahl Tokens stark verringert werden. Insbesondere bei Sprachen wie Deutsch und Französisch, wo es sehr viele Verbformen gibt.

* **stopwords**: Wörter wie *and*, *for* und Artikel haben wenig Aussagekraft und es bietet sich häufig an, diese vor der Tokenisation aus den Texten zu entfernen. Welche Wörter als Stopwords infrage kommen, hängt stark vom Anwendungsfall ab.

Weiter kann die Featurematrix reduziert werden, indem sehr seltene Worte aus dieser entfernt werden. Diese verursachen häufig nur Noise und tragen nichts zur Erklärbarkeit der Daten bei. Auch wenig zur Erklärbarkeit tragen Wörter bei, die sehr häufig in den Texten vorkommen. Insbesondere, wenn sie in jedem einzelnen der Texte auftauchen. Solche Wörter können auch aus der Matrix entfernt werden.

### N-Gramm

Häufig sind Wortkombinationen aussagekräftiger als einzelne Worte. Insbesondere Adjektive können die Bedeutung eines Wortes verändern. So kann ein *movie* *good* oder *bad* sein. Wenn mehrere solche Adjektive im selben Text vorkommen, ist es relevant zu wissen, welches Nomen sie beschreiben. Um diesen Kontext in die Featurematrix aufnehmen zu können, lassen sich sogenannte n-Gramms bilden. $n$ definiert in diesem Fall die Anzahl aufeinanderfolgender Worte, die kombiniert werden sollen. Für jede Kombination wird ein neues Feature erstellt, wofür für jeden einzelnen Text auch wieder der TFIDF-Score berechnet wird. Der Parameter $n$ kann auch als Intervall beschrieben werden, so dass zum Beispiel auch Kombinationen aus einem bis drei Worten möglich sind. Es ist jedoch darauf zu achten, dass je länger eine Kombination ist, desto mehr Tokens wird die resultierende Matrix enthalten. Auch hier empfiehlt es sich, seltene Kombinationen wieder daraus zu entfernen.

### Dimensionsreduktion

Um diese grossen Featurematrizen in ihrer Dimension zu verkleinern, kann PCA verwendet werden. Dadurch werden neue Features gebildet, die die grössten Varianzen in den bestehenden Features abbilden. Es kann dadurch zwar nicht mehr von den neuen Features auf die ursprünglichen Features geschlossen werden, was im Fall von unserem Recommendersystem auch nicht mehr nötig ist.

## Tokens

Um die Tokens zu erstellen, lesen wir die Metadaten zu den Filmen ein und entfernen Filme, die keinen Wert im Attribut `overview` enthalten.
"""

# %%
meta = pd.read_csv('data/movies_meta.csv')

# %%
print('Anzahl Filme mit Wert NA in overview:', meta.overview.isna().sum())

#%%
meta = meta.dropna(subset='overview')
meta.overview.head()

# %% [markdown]
"""
Nun können mit unserer Library-Funktion `tokenize` aus `tfidf.py` die Texte aus dem Attribut `overview` in TFIDF-Tokens umgewandelt werden. Die Funktion verwendet die Library `scikit-learn` und gibt ein DataFrame mit den Tokens als Spalten und den Filmen als Zeilen zurück.

Um die Anzahl an Tokens zu reduzieren, werden wir vor dem Erstellen der Tokens die Texte alle in Kleinbuchstaben umwandeln. Der Nachteil ist, dass Wörter mit denselben Buchstaben, die aber mit einem Grossbuchstaben beginnen, in einigen Fällen eine andere Bedeutung haben, als Wörter mit nur Kleinbuchstaben. Dies ist aber eher ein Problem mit deutschen Texten und sollte relativ selten für englische vorkommen.
"""

# %%
if isNotpdfGenerator:
    tokens = tokenize(meta, stopword=False, lowercase=True)
    print('Anzahl Terme:', tokens.shape[1])

# %% [markdown]
"""
Die resultierende Featurematrix enthält Spalten für über 30'000 Terme. Dies sind ziemlich viele Dimensionen und die Berechnung der Ähnlichkeiten mit dieser würde einige Zeit beanspruchen. Wir wollen nun versuchen die Grösse zu reduzieren. Unser erster Versuch ist es, sogenannte Stopwords zu entfernen, also Wörter die eine sehr geringe bis gar keine Aussagekraft haben (zum Beispiel *the*, *and*, *or*).
"""

# %%
if isNotpdfGenerator:
    tokens = tokenize(meta, stopword=True, lowercase=True)
    print('Anzahl Terme:', tokens.shape[1])

# %% [markdown]
"""
Es scheint nicht allzu viele Stopwords gegeben zu haben. Wir konnten die Anzahl Terme nur um fast 300 reduzieren. Eine weitere Möglichkeit die Anzahl Terme zu reduzieren, ist es eine Mindestanzahl zu definieren. Terme die im gesamten Text seltener vorkommen als diese Anzahl, werden nicht in die Matrix übernommen.
"""

# %%
if isNotpdfGenerator:
    results = []
    for n in range(1, 20):
        tokens = tokenize(meta, stopword=True, lowercase=True, minDf=n)
        results.append({'min': n, 'n': tokens.shape[1]})

# %%
if isNotpdfGenerator:
    results = pd.DataFrame(results)

    plt.figure(figsize = (10, 5))
    plt.plot(results['min'], results['n'])
    plt.title('Anzahl Terme für unterschiedliche Minimumwerte')
    plt.xlabel('Mindestanzahl')
    plt.ylabel('Anzahl Terme')
    plt.xticks(results['min'])
    plt.ylim(0, results['n'].max() + 1000)
    plt.show()

# %% [markdown]
"""
Es zeigt sich, dass mit dem Steigen der Mindestanzahl die Anzahl Terme stark abnimmt. So sind bei mindestens 3 Vorkommnissen nur noch weniger als die Hälfte der Terme vorhanden als bei einer Mindestanzahl von 1. Bei der Einschränkung für mindestens 10 Vorkommnisse, bleiben noch weniger als 5000 Terme übrig.

Bei der Festlegung dieser Grenze gilt es abzuwägen zwischen dem Entfernen von seltenen und irrelevanten Worten und dem Behalten von seltenen, aber aussagekräftigen Begriffen. Genau letztere Attribute sind wichtig für das Recommendersystem, da sie ideal sind, um ähnliche Filme zu erkennen.

Da wir das Risiko klein behalten wollen, hilfreiche Wörter aus der Featurematrix zu enfernen, entscheiden wir uns eine Mindestanzahl von 3 zu verwenden. Die Grösse der Matrix wird dadurch dennoch signifikant verkleinert.
"""

# %% [markdown]
"""
Wir wollen die Matrix weiter verkleinern, indem wir Tokens aus ihr entfernen, die sehr häufig vorkommen. Dazu verwenden wir den `CountVectorizer` von `scikit-learn`, auf welchen auch die TFIDF-Implementierung aufbaut, um die Verteilung der Worthäufigkeiten bestimmen zu können.
"""

# %%
vectorizer = CountVectorizer(stop_words='english', lowercase=True, min_df=3)
word_counts = np.array(vectorizer.fit_transform(meta.overview).sum(axis = 0)).flatten()

# %%
plt.figure(figsize = (10, 5))
plt.hist(word_counts, bins=100)
plt.title('Anzahl Wortvorkommnisse')
plt.xlabel('Anzahl Wortvorkommnisse')
plt.ylabel('Häufigkeit')
plt.show()

# %% [markdown]
"""
Es zeigt sich, dass die Verteilung stark rechtsschief ist und somit die allermeisten Wörter sehr selten vorkommen.
"""

# %%
plt.figure(figsize = (10, 5))
plt.hist(word_counts[word_counts > 500], bins=100)
plt.title('Anzahl Wortvorkommnisse > 500')
plt.xlabel('Anzahl Wortvorkommnisse')
plt.ylabel('Häufigkeit')
plt.show()

# %% [markdown]
"""
Da die Häufigkeit von Wörtern, die mehr als 500 mal vorkommen, nicht erkennbar ist, erstellen wir einen weiteren Plot, in welchen nur diese angezeigt werden. Es scheint tatsächlich nur noch wenige solche Worte zu geben. Möglicherweise handelt es sich hier um Stopwords, die im Standardset von `scikit-learn` fehlen.
"""

# %%
vectorizer.get_feature_names_out()[word_counts > 500]

# %% [markdown]
"""
Wenn wir diese Tokens nun ausgeben, können wir erkennen, dass diese entgegen unserer Erwartung einen informativen Charakter haben. Wörter wie *american*, *documentary* und *school* können für unser Recommendersystem sehr interessant sein. Diese wollen wir behalten und legen deswegen keine Obergrenze bei der Worthäufigkeit fest. Eine Reduktion dadurch wäre aufgrund der vielen seltenen Wörter sowieso nur sehr gering ausgefallen.
"""

# %%
if isNotpdfGenerator:
    tokens = tokenize(meta, stopword=True, lowercase=True, minDf=3)
    tokens.to_csv('data/movies_tfidf.csv')

# %% [markdown]
"""
Wir speichern die generierten Tokens nun in der Datei `data/movies_tfidf.csv` ab, so dass wir sie für die Recommendersysteme verwenden können.

## Modellierung des Recommendersystems

Die Featurematrix ist nun bereit und wir können diese verwenden, um unseren ersten Recommendersysteme mit diesen Daten zu bauen.
Wir erreichen mit dieser Feature-Matrix eine Precision von knapp **0.585**, diese ist deutlich tiefer als bei unserem Baseline Modell.
"""

# %%
#tokens = pd.read_csv('data/movies_tfidf_noDR.csv',index_col="movieId")

config = CakeConfig(
    {MatrixGenerator.CONST_KEY_TFIDF: np.array(1)},
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE,
)

resultTFIDF = []

# Init Recommender
matrixTfidf = MatrixGenerator(tfidfTokens='data/movies_tfidf.csv')
baseCakeommenderCosine = Cakeommender(
    "TFIDFCakeommender",
    config,
    matrixTfidf,
    verbose=True
)

eval = Evaluation("TFCake",config,matrixTfidf)
precision = eval.precision()
topNPrecision = eval.topNPrecision()
novelty = eval.novelty()
resultTFIDF.append({
    "n_grams": "keine",
    "Singulärwerte": 0,
    "precisionMean": precision[0],
    "precisionStd": precision[1],
    "novelty": novelty,
    "topNPrecision": topNPrecision
})

# %% [markdown]
"""
Um die Rechenzeit zu minimieren, versuchen wir die Matrix zu komprimieren mittels SVD. Die Recommendersysteme werden mit unterschiedlicher Anzahl von Singulärwerten (10, 100, 1000) erstellt und evaluiert. Die Ergebnisse der Precision sind durchwegs schlechter.

"""

# %%
def cakeWithSvd(token: pd.DataFrame, singularvalues: int)-> tuple():
    """
    Jan
    Evaluation for TF-IDF models with a SVD komprimation.

    :param pd.DataFrame token: TF-IDF Tokens
    :param int singularvalues: TF-IDF Tokens
    :return : Precisions of 5 folds, standartdeviation, novelty and Top-10 Precision
    :rtype : tuple()
    """
    filename = f'data/movies_tfidf_{singularvalues}d.csv'

    svd = TruncatedSVD(n_components=singularvalues)
    reduced_tokens = svd.fit_transform(token)
    reduced_tokens = pd.DataFrame(reduced_tokens, index=token.index)
    reduced_tokens.to_csv(filename)

    config = CakeConfig(
        {MatrixGenerator.CONST_KEY_TFIDF: np.array(1)},
        SimilarityEnum.COSINE,
        RatingScaleEnum.TERTIARY,
        FeatureNormalizationEnum.ZSCORE,
    )

    matrixTfidf = MatrixGenerator(tfidfTokens=filename)
    eval = Evaluation(f"TFCake_{filename}d", config, matrixTfidf)

    precision = eval.precision()
    novelty = eval.novelty()
    topNPrecision = eval.topNPrecision()
    return precision, novelty, topNPrecision

for singularValues in [10, 100, 1000]:
    precision, novelty, topNPrecision = cakeWithSvd(tokens, singularValues)
    resultTFIDF.append({
        "n_grams": "keine",
        "Singulärwerte": singularValues,
        "precisionMean": precision[0],
        "precisionStd": precision[1],
        "novelty": novelty,
        "topNPrecision": topNPrecision
    })

filename = 'data/movies_tfidf_1-4grams.csv'
if isNotpdfGenerator:
    tokens = tokenize(meta, stopword=True, lowercase=True, minDf=3, ngramRange=(1,4))
    tokens.to_csv(filename)

config = CakeConfig(
    {MatrixGenerator.CONST_KEY_TFIDF: np.array(1)},
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE,
)

matrixTfidf = MatrixGenerator(tfidfTokens=filename)
eval = Evaluation("TFCake_1-4grams", config, matrixTfidf)

precision = eval.precision()
novelty = eval.novelty()
topNPrecision = eval.topNPrecision()
resultTFIDF.append({
    "n_grams": "(1-4)",
    "Singulärwerte": 0,
    "precisionMean": precision[0],
    "precisionStd": precision[1],
    "novelty": novelty,
    "topNPrecision": topNPrecision
})

# %% [markdown]
"""
In einem nächsten Schritt versuchen wir uns mit dem n-grams Ansatz von 1-4 Kombinationen. Die Precision ist immer noch leicht unter dem nicht reduzierten TF-IDF Modell. Da aufgrund des n-grams Ansatzes mehr Tokens hinzugekommen sind, werden diese nochmals mittels SVD reduziert. Eine Kombination von SVD mit 6000 Singulärwerten und einem n-grams Ansatz verschlechtert die Precision nochmals.
"""

# %%
precision, novelty, topNPrecision  = cakeWithSvd(tokens, 6000)
resultTFIDF.append({
    "n_grams": "(1-4)",
    "Singulärwerte": 6000,
    "precisionMean": precision[0],
    "precisionStd": precision[1],
    "novelty": novelty,
    "topNPrecision": topNPrecision
})

# %% [markdown]
"""
Die besten Tokens werden exportiert. Das beste TF-IDF Modell ist jedoch nicht besser wie das Baseline-Modell.
"""

# %%
resultTFIDF = pd.DataFrame(resultTFIDF)

fig = plt.figure(figsize=(20,10),dpi = 80)
gs = fig.add_gridspec(4, 2, hspace= 0.4)
# subplot Precision
x = "Singulärwerte: " + resultTFIDF["Singulärwerte"].astype(str) + " | n_grams: " + resultTFIDF["n_grams"]
y = resultTFIDF["precisionMean"]
y_error = resultTFIDF["precisionStd"]

fig.suptitle("Quantitative Evaluierung von SVD und n-grams",fontsize = 20)
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
sns.scatterplot(x='Singulärwerte', y='topNPrecision', ax=topNPrecision, data=resultTFIDF, hue='n_grams')
topNPrecision.set_title("Top-10 Precision")
topNPrecision.set_ylabel("Precision")

# subplot Novelty
novelty = fig.add_subplot(gs[2:4, 1])
novelty.set_title("Top-10 Novelty")
sns.scatterplot(x='Singulärwerte', y='novelty', ax=novelty, data=resultTFIDF, hue='n_grams')
plt.show()

# %%
%reset -f
