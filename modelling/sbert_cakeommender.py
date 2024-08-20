# %%
import os
if os.getcwd().endswith('modelling'):
    os.chdir('..')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from libraries.bert import Bertomat,BertEnum
from libraries.pipeconfig import (
    CakeConfig,
    FeatureNormalizationEnum,
    SimilarityEnum,
    RatingScaleEnum,
)
from libraries.matrix import MatrixGenerator
from cakeommender import Cakeommender
from evaluation import Evaluation, plot_results
from hdbscan import HDBSCAN
from umap import UMAP
from scipy.stats import pearsonr
from libraries.feature_evaluation import plotSimiliarities, plotHDBSCANProbabilities, plotHDBSCANClustersAlongGenres, plotUMAPEmbeddingsAlongGenres, plotCumulativePCAVariance, plotUMAPEmbeddingsGenreAlongTopNGenres
from sklearn.decomposition import PCA

# %% [markdown]
"""
# Recommendersystem mit Sentence-BERT Sentence Embeddings

Bei diesen Recommendersystemen wollen wir Sentence Embeddings von SBERT als Featurematrix verwenden. Wir wandeln dazu den Text `overview` in Tokens um und generieren daraus die Embeddings mit einem vortrainierten SBERT-Modell.

## Steckbrief
SBERT (Sentence-Bidirectional Encoder Representations from Transformers) ist eine Modifikation basierend auf BERT. SBERT kann wie BERT für verschiedene NLP-Tasks verwendet werden.

### Ähnlichkeiten mit BERT

BERT ist der de facto Standard für Text-Token-Generation und die daraus resultierenden Embeddings.
Diese Embeddings können für diverse NLP-Tasks verwendet werden.

Unser Baseline-Recommendersystem berechnet Ähnlichkeiten von Filmen aufgrund von Metadaten wie bsp. "Spieldauer".
Der NLP Task unseres Recommendersystems ist also das Vergleichen von bsp. Filmbeschreibungen und das Generieren von Ähnlichkeiten zwischen den verschiedenen Filmen bzw. Beschreibungen.
Dazu müssen die Beschreibungen vergleichbar gemacht werden.

### BERT Hindernisse

Um dies in BERT umzusetzen müssten wir den folgenden Ansatz durchführen:
Die zu vergleichenden Sätze müssen, mit einem [SEP] Token dazwischen, zusammengehängt werden.
Diese Kombination kann dann in das BERT-Modell eingespiesen werden und es würde die Ähnlichkeit dieser zwei Sätze liefern.
Um mit diesem Verfahren die jeweils ähnlichsten Filme für alle Filme finden zu können, müssten wir alle möglichen Paarkombinationen der Beschreibungen in das BERT-Modell geben und berechnen lassen.
Danach haben wir alle Ähnlichkeitswerte und können danach sortieren.
Für `n` Filme/Beschreibungen resultiert dies in einer Komplexität von `n(n-1)/2`. Für unser Recommendersystem mit ca. 10'000 Filmen bedeutet dies **49'995'000** Durchgänge, was zu viel Zeit beanspruchen würde als das man dieses System produktiv einsetzen könnte.

Um dieses Problem zu umgehen, haben wir für unsere BERT-Recommendersysteme Sentence-Embeddings mit den BERT-Modellen generieren lassen. Wir haben dazu einerseits die Embeddings für das *[CLS]*-Token verwendet oder auch den Mittelwert über alle Word-Embeddings übernommen. Wie wir gesehen haben, konnte damit kein brauchbares Recommendersystem gebaut werden.

### SBERT

SBERT löst diese BERT-Hindernisse indem diese Sentence-Embeddings besser berechnet werden. Das Hauptproblem der Kombinierung und Embeddings Berechnung für Satzpaare entfällt dabei.

Die Architektur von SBERT ist ein sogenanntes `Twin-Network`. Diese Netze können zwei Datenpunkte (Sätze) gleichzeitig und auf gleiche Art verarbeiten.
Auch die Weights und Biases sind dieselben bei der Verarbeitung, somit kann man sich vorstellen das dies ein einzelnes Modell ist, welches mehrfach angewendet wird.
<br><br>
<center>

![Zwillings Netzwerk Architektur von SBERT](modelling/res/SBERT.png)

</center>
<br><br>

Auf dem Bild, welches aus dem Paper von **Reimers & Gurevych** stammt, ist zu erkennen, dass der Hauptbestandteil im neuen SBERT-Modell immer noch BERT ist. Die einzige Ergänzung dabei ist ein neuer Pooling-Layer nach dem BERT-Modell.
Pooling-Layers werden genützt um die Dimensionen zu reduzieren.
Dieser Pooling-Layer erlaubt es uns durch das Mitteln aller BERT-Output-Vektoren, ein Embedding für einen ganzen Satz zu generieren, welches besser geeignet ist, um dessen Ähnlichkeiten berechnen zu können.


### Referenzen

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. https://doi.org/10.48550/ARXIV.1908.10084
- Karlsson, V. (2020, July 11). SentenceBERT — Semantically meaningful sentence embeddings the right way. DAIR.AI. https://medium.com/dair-ai/tl-dr-sentencebert-8dec326daf4e




## Berechnen der Sentence Embeddings

Zuerst müssen die Metadaten aus der CSV-Datei eingelesen werden und Filme ohne Text im Attribut `overview` entfernt werden. Für diese können wir keine Embeddings generieren und sind deshalb für unser Recommendersystem uninteressant.
"""

# %%
meta = pd.read_csv(r'./data/movies_meta.csv')

# %% [markdown]
"""
Es sind dieselben zehn Filme wie schon bei den anderen Modellen.
"""
# %%
print('Anzahl Filme mit Wert NA in overview:', meta.overview.isna().sum())
meta = meta.dropna(subset = 'overview')
meta.shape

# %%
n = 10
# %%
sbertomat = Bertomat(BertEnum.SBERT)
sentence_embeddings = pd.read_csv('data/movies_SBert.csv', index_col='movieId')
# sentence_embeddings = sbertomat.generateEmbeddings(meta[0:n])
sentence_embeddings.shape

# %% [markdown]
"""
## Evaluierung Embeddings

Auch diese Sentence-Embeddings wollen wir untersuchen bevor wir mit diesen Recommendersysteme erstellen.
"""

# %%
genres = pd.read_csv('data/movies_genres.csv', index_col='movieId')
movieIds = genres.index.intersection(sentence_embeddings.index)
sentence_embeddings = sentence_embeddings.loc[movieIds].sort_index()
genres = genres.loc[movieIds].sort_index()
genres.shape

# %% [markdown]
"""
Die Ähnlichkeiten zwischen den Filmen sind mit SBERT nun annähernd normalverteilt. Die BERT-Embeddings waren viel stärker linksschief verteilt, wobei es sehr viele Ähnlichkeiten bei 1.0 gab.
"""

# %%
fig, embeddings_sim = plotSimiliarities(sentence_embeddings, 'Embeddings', return_sim=True)

# %% [markdown]
"""
Zum Vergleich hier nochmals die Ähnlichkeiten zwischen den Filmen, wenn Genres als Features verwendet werden.
"""

# %%
fig, genres_sim = plotSimiliarities(genres, 'Genres', return_sim=True)

# %% [markdown]
"""
Der Pearson-Korrelationskoeffizient zwischen Genre- und Embedding-Ähnlichkeiten liegt hier bei knapp **0.2**. Für die BERT-Embeddings lag dieser noch ca. zehnmal tiefer. Die Korrelation ist hier stärker, wenn auch nicht sonderlich stark. Es wäre aber möglich, dass diese Embeddings nun näher bei den Genres sind.
"""

# %%
pearsonr(embeddings_sim.flatten(), genres_sim.flatten())

# %% [markdown]
"""
Auch mit diesen Sentence-Embeddings wollen wir ein Clustering vornehmen. Wir wählen das gleiche Vorgehen wie bei den BERT-Embeddings und starten mit `HDBSCAN`. Es resultieren drei Cluster, was im Vergleich zu den Genres relativ wenige sind.
"""

# %%
hdbscan = HDBSCAN().fit(sentence_embeddings)
print('Anzahl gefundene Cluster:', hdbscan.labels_.max() + 1)

# %% [markdown]
"""
Eine Mehrheit der Filme wurde keinem Cluster zugewiesen, was man an der Wahrscheinlichkeit von 0 erkennen kann. Dafür besitzen die Filme, die einem Cluster zugewiesen wurden, eine sehr hohe Chance, dass sie auch wirklich zu diesem gehören.
"""

# %%
fig = plotHDBSCANProbabilities(hdbscan)

# %% [markdown]
"""
Ungefähr zwei Drittel der Filme konnten keiner Gruppe zugewiesen werden. Das grösste Cluster deckt etwa das restliche Drittel ab. Nur einzelne Filme wurden den zwei weiteren Cluster zugeschrieben.
"""

# %%
pd.Series(hdbscan.labels_).value_counts()

# %% [markdown]
"""
Wenig überraschend lässt sich bei diesem einen Cluster und den ungruppierten Filmen kein Muster erkennen, wenn man diese den Genres gegenüberstellt. Entweder gibt es keine oder eine nur sehr geringe Übereinstimmung zwischen Embeddings und Genres, oder `HDBSCAN` kann aufgrund seiner Funktionsweise keine Übereinstimmung feststellen. Eine andere Parameterisierung des Algorithmus bringt leider keine Verbesserung.
"""

# %%
fig = plotHDBSCANClustersAlongGenres(hdbscan, genres)

# %% [markdown]
"""
Aus diesem Grund wollen wir nun noch UMAP einsetzen, in der Hoffnung, dass mit diesem Algorithmus womöglich Muster erkannt werden können.
"""

# %%
umap = UMAP(min_dist=0.0, n_neighbors=10)
umap_embeddings = umap.fit_transform(sentence_embeddings)

# %% [markdown]
"""
Wenn wir die zweidimensionale Repräsentation der Embeddings über die verschiedenen Genres plotten, können wir dieses Mal Ansammlungen von Filmen bei den Genres erkennen. Je nach Genre werden unterschiedliche Wertebereiche auf den beiden Komponenten abgebildet. Dokumentationsfilme tendieren eher Werte auf der unteren Hälfte der Skala von Komponente 1 abzudecken, wohingegen sich Horror- und Thriller-Filme eher auf der oberen Hälfte des Wertebereichs befinden. Diese beiden Genres sind auch eher ähnlich, somit scheint eine ähnliche Platzierung auch sinnvoll. Diese meist fiktionalen Genres unterscheiden sich stark von den nicht-fiktionalen Dokumentarfilmen, was die gegenteilige Anordnung auch erklären mag. Natürlich gibt es bei allen Genres Ausreisser. Dies mag einerseits damit zu tun haben, dass die Embeddings eben doch mehr Informationen als nur die Genres beinhalten und andererseits damit, dass die meisten Filme mehrere Genres zugewiesen haben.

Im Gegensatz zu den BERT-Embeddings sind bei den Embeddings von SBERT klar Muster zu erkennen.
"""

# %%
fig = plotUMAPEmbeddingsAlongGenres(umap_embeddings, genres)

# %% [markdown]
"""
Nachfolgend wollen wir nun einzelne Genres genauer anschauen. Da Filme mehreren Genres zugeordnet werden können, wollen wir ein einzelnes Genre mit den Top-3 Genres vergleichen, denen am meisten Filme des Hauptgenres zuegordnet wurden. Die folgenden Plots zeigen jeweils die UMAP-Projektion für Filme des Hauptgenres und der drei weiteren meistzugewiesenen Genres. Dabei werden jeweils die anderen Genres eingefärbt.

Wenn wir nun nur den Plot für das Genre Horror betrachten, können wir sehen, dass Filme, die auch als Thriller eingeteilt wurden, das ganze Horror-Spektrum abdecken. Die Genres scheinen also tatsächlich sehr ähnlich zu sein. Horrorfilme tendieren aber dazu Werte grösser als 10 auf der Komponente 1 anzunehmen. Dies können wir an der Verteilung bei Thriller, Drama und Comedy erkennen, für die es auch einige Filme unterhalb dieses Schwellwerts gibt. Die Horror-Filme sammeln sich abgesehen von einigen Ausreissern bei Drama und Comedy in lokalen Bereichen.
"""

# %%
fig = plotUMAPEmbeddingsGenreAlongTopNGenres(umap_embeddings, genres, main_genre='Horror', n=3)

# %% [markdown]
"""
Wenn wir zum Genre Music wechseln, bei dem die Top-3 Genres auch aus Drama und Comedy besteht, sehen wir, dass sich hier die Music-Filme klar im unteren Bereich der Komponente-1-Skala befinden. Die Ansammlung von Filmen um 6 (Komponente 1) und zwischen -2 und 0 (Komponente 2) bei Music, ist bei den drei anderen Genres ganz klar zu erkennen und wird durch Music-Filme dominiert. Documentary Drama und Comedy sind viel breiter verteilt als Music, und Filme aus letzterem Genre befinden sich jeweils nur in einem kleinen Bereich der anderen Genres. Auch hier lassen sich also interessante Strukturen erkennen.
"""

# %%
fig = plotUMAPEmbeddingsGenreAlongTopNGenres(umap_embeddings, genres, main_genre='Music', n=3)

# %% [markdown]
"""
Wir fahren mit dem Documentary-Genre fort. Hier ist Music das Top-Genre. Dokumentationsfilme, aus dem Bereich Music scheinen sich eher im unteren Teil der Plots zu befinden. Filme aus den anderen beiden Top-3 Genres (History und Drama) sind tendenziell eher bei höheren Werten der Komponente 2 zu finden. Es scheint auch hier leichte Abgrenzungen zu geben.

Diejenigen Music-Filme, die auch Dokumentationen sind, sind dabei im oberen Bereich der Plots und die anderen eher weiter unten.
"""

# %%
fig = plotUMAPEmbeddingsGenreAlongTopNGenres(umap_embeddings, genres, main_genre='Documentary', n=3)

# %% [markdown]
"""
Nachdem wir die Embeddings nun analysiert haben, speichern wir diese nun als Featurematrix ab.
"""

# %%
sentence_embeddings.to_csv('data/movies_sbert.csv')

# %% [markdown]
"""
## Modellierung des Recommendersystems

Da wir nun die Embeddings etwas besser kennen, können wir diese verwenden, um unser erstes Recommendersystem mit diesen Daten zu bauen. Auch hier verwenden wir wieder die tertiäre Skala, da sie die besten Resultate im Baseline-Recommendersystem geliefert hat.
"""

# %%
config = CakeConfig(
    {MatrixGenerator.CONST_KEY_SBERT: np.array(1)},
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE,
)

matrix = MatrixGenerator(sbertEmbeddings=True)

eval = Evaluation('SBERTCakeommender', config, matrix)
mean_precision, std_precision = eval.precision()
print(f'Precision: {mean_precision.round(3)} (+/- {std_precision.round(3)})')

# %% [markdown]
"""
### Dimensionsreduktion mit PCA

Durch eine Dimensionsreduktion könnte das Modell möglicherweise noch weiter verbessert werden, da unwichtige Details entfernt werden. Ausserdem würde es die Berechnung der Ähnlichkeiten beschleunigen. Wie schon beim BERT-Recommendersystem testen wir nun wieder unterschiedliche Dimensionen für die Featurematrix und vergleichen anschliessend die Metriken.
"""

# %%
results = [{
    'dim': 768,
    'mean_precision': mean_precision,
    'std_precision': std_precision,
    'topn_precision': eval.topNPrecision(),
    'novelty': eval.novelty()
}]

for dim in [2, 5, 10, 100, 200, 500]:
    pca = PCA(n_components=dim)
    reduced_embeddings = pca.fit_transform(sentence_embeddings)
    reduced_embeddings = pd.DataFrame(reduced_embeddings, index=sentence_embeddings.index)

    embeddings_file_reduced = f'data/movies_sbert_{dim}d.csv'
    reduced_embeddings.to_csv(embeddings_file_reduced)

    matrix = MatrixGenerator(sbertEmbeddings=embeddings_file_reduced)
    eval = Evaluation(f'SBERT_{dim}d', config, matrix)
    mean_precision, std_precision = eval.precision()
    results.append({
        'dim': dim,
        'mean_precision': mean_precision,
        'std_precision': std_precision,
        'topn_precision': eval.topNPrecision(),
        'novelty': eval.novelty()
    })

# %% [markdown]
"""
Durch die Reduktion auf zwei Dimensionen liefert das Modell nur noch so gute Werte wie das Modell mit Zufallsdaten. Hier scheint zuviel Informationen verloren gegangen zu sein. Erstaunlich ist, dass bereits mit fünf PCA-Komponenten nahezu ähnlich gute Resultate erzielt werden können, wie mit den ursprünglichen Sentence-Embeddings. Bei der Top-10-Precision ist das Modell sogar deutlich besser als mit der vollständigen Embedding-Featurematrix. Weitere Komponenten können das Modell nicht mehr erhöhen und es werden sogar niedrigere Werte für die Metriken erreicht.
"""

# %%
results_df = pd.DataFrame(results).sort_values('dim')
fig = plot_results(
    results_df.dim.astype(str) + 'D', results_df.mean_precision, results_df.std_precision, results_df.topn_precision, results_df.novelty, 'Dimensionen'
)

# %% [markdown]
"""
Dies ist insofern erstaunlich, dass mit fünf Komponenten nur ca. 15 % Varianz abgedeckt wird. In diesen Bruchteil scheinen jedoch die wichtigen Informationen für unser Recommendersystem vorhanden zu sein.
"""

# %%
pca = PCA().fit(sentence_embeddings)
print(f'Enthaltene Varianz mit 5 Komponenten: {round(pca.explained_variance_ratio_[0:5].sum() * 100, 2)} %')

# %% [markdown]
"""
Um tatsächlich 95 % der Daten zu behalten, wären knapp 300 Komponenten nötig.
"""

# %%
fig, _ = plotCumulativePCAVariance(pca)

# %%
%reset -f
