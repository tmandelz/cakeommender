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

from libraries.matrix import MatrixGenerator, MatrixLoader
from libraries.weights import WeightGenerator
from cakeommender import Cakeommender
from evaluation import Evaluation


# %% [markdown]
"""
# Kombiniertes Recommendersystem

Nachdem wir nun einige Recommendersysteme erstellt haben, wollen wir die erhaltenen Erkenntnisse nutzen, um durch Kombination der verschiedenen Featurematrizen noch bessere Modelle zu erhalten. Da unser Baseline-Recommendersystem auf Basis der herkömmlichen Features am besten abgeschnitten hat, dient es hier als Basis und wollen es mit einem NLP-Ansatz weiter verbessern. In dieser Kategorie hat SBERT die besten Werte erzielt, weswegen wir dessen Embeddings verwenden werden. Die kombinierte Featurematrix ist ziemlich gross und nicht alle Features erscheinen uns gleich wichtig. Deswegen variieren wir die Gewichtungen der Feature-Gruppen, so dass ein möglichst gutes Modell erreicht wird.

"""
# %%
matrixBaseSbert = MatrixGenerator(metadata=True, genres=True, actors=True, directors=True, sbertEmbeddings='data/movies_sbert_5d.csv')
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

# %% [markdown]
"""
Nach unterschiedlichsten getesteten Gewichtungen konnte eine leichte  Verbesserungen gefunden werden. Somit benötigen wir für das beste Modell, was die Metriken betrifft, eine Kombination von Baseline und SBERT-Modell.
"""

# %%
eval = Evaluation('Baseline_SBERT', config, matrixBaseSbert)
novelty = eval.novelty()
print('Novelty ' + str(round(novelty, 1)))
precisionMean, precisionStd = eval.precision()

topNPrecision = eval.topNPrecision()
print("Top-10 Precision " + str(round(topNPrecision, 2)))

plt.figure(figsize = (10, 5))
plt.errorbar(['model'], precisionMean, yerr = precisionStd, fmt ='o')
plt.title('Precision Baseline + SBERT Recommendersystem')
plt.ylabel('Precision')
plt.show()

# %% [markdown]
"""
# Überprüfung des Modelles anhand von Vorschlägen

Um unser bestes Recommendersystem nun nochmals qualitativ zu überprüfen, werden ausgewählte Filme direkt als Userprofil verwendet und generieren daraus Empfehlungen. Wir wollen dabei untersuchen, was passiert wenn die SBERT-Embeddings stärker oder schwächer gewichtet werden.
"""

# %%
matrixBase = MatrixGenerator(metadata=True, genres=True, actors=True, directors=True)
config = CakeConfig(
    {
        MatrixGenerator.CONST_KEY_METADATA: np.array(1),
        MatrixGenerator.CONST_KEY_GENRES: np.array(1),
        MatrixGenerator.CONST_KEY_ACTORS: np.array(0.4),
        MatrixGenerator.CONST_KEY_DIRECTORS: np.array(0),
    },
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE
)

bestModel = Cakeommender("Best_Baseline", config, matrixBase)

movies = pd.read_csv('./data/movies.csv', low_memory=False)[["movieId","original_title"]]

def predictTopNForMovie(partOfTitle: str, data: pd.DataFrame=movies, model: Cakeommender=bestModel) -> None:
    """
    Jan Zwicky
    print top-10 movies

    :param str partOfTitle: Part of a Title of a movie
    :param pd.DataFrame data: movies Data
    :param Cakeommender model: model
    """
    filteredMovies = data.loc[data["original_title"].str.contains(partOfTitle, na=False), :].iloc[0]

    model.calcAppUserProfiles([[filteredMovies["movieId"]]])
    recommendations = model.predictTopNForUser(users=["0"], n=10, removeRatedMovies=False)
    recommendations = pd.merge(data, recommendations, on="movieId", how="inner")["original_title"]
    recommendations.index = recommendations.index + 1

    print(f'Filmempfehlungen für Film {filteredMovies["original_title"]} von {model.name}:')
    print(recommendations)

# %% [markdown]
"""
Wenn wir uns die Top-10-Empfehlungen für den Film *Inception* für das Baseline- und das SBERT-Recommendersystem anschauen, sehen wir, dass die Vorschläge vollkommen unterschiedlich ausfallen. Sie scheinen aber in beiden Fällen plausibel und gut zu sein. Das Modell mit SBERT-Embeddings schlägt Filme sehr stark anhand dessen vor, was im Film passiert. Das Baseline-System fokussiert sich stärker auf Genres und Schauspieler.
"""

# %%
matrixSbert = MatrixGenerator(sbertEmbeddings='data/movies_sbert_5d.csv')
config = CakeConfig(
    {
        MatrixGenerator.CONST_KEY_SBERT: np.array(1)
    },
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE
)
sbertModel = Cakeommender("SBERT", config, matrixSbert)

# %%
predictTopNForMovie("Inception", model=bestModel)

# %%
predictTopNForMovie("Inception", model=sbertModel)

# %% [markdown]
"""
## Kombination

Nun wird nochmals versucht eine Kombination zwischen den beiden Recommendersystemen zu finden. Es sollten ähnlich viele Filme vom Baseline-, wie vom SBERT-Recommendersystem vorgeschlagen werden.
"""

# %%
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
baseSbertModel = Cakeommender("Baseline_SBERT", config, matrixBaseSbert)

# %% [markdown]
"""
Eine Kombination bestehend aus 60 % SBERT-Embeddings und den gleich bleibenden Gewichtungen des Baseline-Teils, ergibt einigermassen ausgeglichene Vorschläge, wie an den folgenden Top-10-Listen erkannt werden kann. Für Iron Man 2 werden wie erwartet diverse Marvel-Filme vorgeschlagen und für den Harry Potter-Film werden andere Filme aus der Reihe empfohlen, sowie Filme mit Bezug zu London, wo dieser Film spielt.
"""

# %%
predictTopNForMovie("Inception", model=baseSbertModel)

# %%
predictTopNForMovie("Iron Man 2", model=baseSbertModel)

# %%
predictTopNForMovie("Ted", model=baseSbertModel)

# %%
predictTopNForMovie("Harry Potter", model=baseSbertModel)

# %% [markdown]
"""
## Fazit

Die Grundlage eines content-based Recommendersystems sind dessen Features, aufgrund von welchen die Ähnlichkeiten zwischen den Filmen (bzw. Nutzern) berechnet werden können. Im Falle von Filmen ist es schwierig diese genug fein zu charakterisieren. Es existieren zwar Metadaten und Genres, aber diese beschreiben den Film nur oberflächlich und die Handlung des Films kann daraus nicht extrahiert werden. Andere Eigenschaften wie Drehorte, Aufbau und Dramaturgie sind für die meisten Filmen nicht verfügbar und wenn, dann können sie nicht direkt als Features verwendet werden. Doch scheinen uns diese Informationen besonders relevant zu sein, um Empfehlungen zu Filmen erstellen zu können. Genres können ein guter Indikator sein, um zu beschreiben, welche Filme ein User mag. Da es aber unzählige Filme zu einem Genre gibt und in der Regel nicht nur ein einzelnes Genre von einem User präferiert wird, kann man die Filmauswahl so nur leicht einschränken.

Was für die allermeisten Filme zur Verfügung steht, ist eine kurze Beschreibung, die einige Schlüsselwörter zum Film enthält. Diese ist zwar kurz, doch kann anhand dieser mehr über den Film bestimmt werden als nur über dessen Genres. Aus diesem Grund erscheint es sinnvoll diese als Grundlage für ein Recommendersystem zu verwenden. Um diese Beschreibung in Zahlen und somit für das Modell brauchbare Features umzuwandeln, bietet sich Natural Language Processing an. Beim Testen von unterschiedlichen NLP-Algorithmen und -Modellen ist uns aber aufgefallen, dass sich nicht alle für diese Aufgabe eignen. Im Falle von TFIDF werden die Featurematrizen sehr schnell sehr gross, was unter anderem die Berechnung der Ähnlichkeiten stark verlangsamt. Zudem kann damit auch kein besonders gutes System erstellt werden. Mit Sentence-Embeddings von BERT kann, unabhängig von unseren getesteten Berechnungsarten, auch kein deutlich besseres Recommendersystem modelliert werden. Dies war jedoch zu erwarten, da sich diese Embeddings nicht für die Berechnung von Ähnlichkeiten zueinander eignen. Die fehlende Struktur konnten wir auch in unserer Embedding-Evaluierung feststellen.

Speziell für unseren Anwendungsfall, dem Berechnen von Cosine-Ähnlichkeiten zwischen Embeddings, wurde das SBERT-Modell entwickelt, mit wessen Sentence-Embeddings wir auch mit Abstand die besten Resultate von allen NLP-Systemen erzielt haben. Dessen Precision-Werte waren jedoch immer noch leicht schlechter als die unseres Baseline-Modells auf Basis von Metadaten. Insgesamt konnten bei allen Systemen Werte für die Metriken erzielt werden, die nur leicht besser waren als beim Modell mit Zufallsfeatures. Bei Betrachtung der tatsächlichen Empfehlungen konnten wir jedoch feststellen, dass die Modelle trotz der mittelmässigen quantitativen Evaluierungsresultate doch sinnvolle Filmvorschläge machen. Dabei war auch zu erkennen, dass das Modell auf Basis von Metadaten und das Modell auf Basis der Filmbeschreibung andere Filme empfehlen. Während das erstgenannte vor allem auf die Genres abzielte, schlägt das NLP-Recommendersystem anhand des Kontexts Filme vor. Dabei werden auch Filme aus der gleichen Reihe und des gleichen Studios empfohlen, was man nicht so gezielt aus den blossen Metadaten und Genres erkennen kann.

Aus diesem Grund erscheint es uns sinnvoll, die beiden Ansätze miteinander zu kombinieren, um die Vorschläge auch etwas zu diversifizieren. Mit dem kombinierten Modell aus Metadaten und SBERT-Embeddings konnten die quantitativen Messwerte einerseits leicht verbessert werden und andererseits scheinen die Filmempfehlungen dadurch weniger eintöntig zu sein. Es werden so nicht mehr nur irgendwelche Filme mit gleichen Genres empfohlen oder alle Fortsetzungen eines Films. Die beiden Strategien werden dadurch kombiniert und es entstehen ansprechende Empfehlungen.

Dass die meisten Modelle bei der quantitativen Evaluierung zwar relativ hohe, aber doch sehr ähnliche Werte erzielt haben, könnte mit unserer Definition eines guten Films zu tun haben. Wir definierten, dass eine Bewertung als positiv gilt, wenn sich die Bewertung in den oberen 40 % des Bewertungsbereichs des Nutzers befindet. Diese dynamische Grenze pro User erschien uns sinnvoll, um auf das spezifische Bewertungsverhalten der Nutzenden einzugehen, da es durchaus kritischere und wohlwollendere Menschen gibt. Dieses Verfahren funktioniert jedoch weniger gut, wenn Nutzende ausschliesslich Bewertungen im oberen oder im unteren Bereich der Ratingskala abgegeben haben. Dann werden diese positiven resp. negativen Bewertungen in positiv und negativ eingeteilt, was nicht in der Absicht des Bewertenden gewesen sein mag. Da die Precision auf dieser positiv-negativ-Information aufbaut, ist es gut möglich, dass diese durch diesen Umstand weniger aussagekräftig ist. Bei Fortführung dieses Projekts, wäre es sinnvoll diesen Rating-Split nochmals zu überdenken und zu überarbeiten.

Beim Testen der unterschiedlichen Konfigurationsmöglichkeiten des Recommendersystems sind in den meisten Fällen keine grossen Unterschiede erkennbar gewesen. Erstaunt hat uns aber, dass die tertiäre Ratingskala am besten abgeschnitten und sogar die standardisierte übertroffen hat. Dass diese besser funktioniert als die binäre, haben wir erwartet, da bei der binären nicht zwischen negativ bewerteten und gar nicht bewerteten Filmen unterschieden werden kann. Wir vermuteten aber, dass die drei Stufen der tertiären Skala dennoch zu ungenau seien und erst durch die kontinuierliche standardisierte Skala genug Informationen vorhanden sind, um ein ausgefeiltes Userprofil erstellen zu können. Die tertiäre Skala scheint aber ausreichend Informationen zu enthalten und womöglich leidet die standardisierte Skala unter zuviel Noise, also kleineren Schwankungen bei den Bewertungen, die das Userprofil ungenauer machen.

Unsere Empfehlung für ein optimales Recommendersystem bei unserem untersuchten Anwendungsfall besteht aus Film-Metadaten und einer Filmbeschreibung, die mittels SBERT in Embeddings transformiert wurde. Dieses Modell möchten wir im nächsten Schritt in einen Prototyp integrieren. Dabei testen wir auch wie es sich verhält, wenn zwei Userprofile kombiniert werden, damit für beide gemeinsame Empfehlungen erstellt werden können.
"""

# %%
%reset -f