# %%
import os
if os.getcwd().endswith('modelling'):
    os.chdir('..')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from libraries.bert import Bertomat, BertEnum
from libraries.pipeconfig import (
    CakeConfig,
    FeatureNormalizationEnum,
    SimilarityEnum,
    RatingScaleEnum,
)
from libraries.matrix import MatrixGenerator
from cakeommender import Cakeommender
from evaluation import Evaluation, plot_results
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
from umap import UMAP
from scipy.stats import pearsonr
from libraries.feature_evaluation import plotSimiliarities, plotHDBSCANProbabilities, plotHDBSCANClustersAlongGenres, plotUMAPEmbeddingsAlongGenres, plotCumulativePCAVariance

# %% [markdown]
"""
# Recommendersystem mit BERT Sentence Embeddings

Bei diesen Recommendersystemen wollen wir Sentence Embeddings von BERT als Featurematrix verwenden. Wir wandeln dazu den Text `overview` in Tokens um und generieren daraus die Embeddings mit einem vortrainierten BERT-Modell.

## Steckbrief

BERT (Bidirectional Encoder Representations from Transformers) ist eine Transformers-Modellarchitektur, die für unterschiedlichste NLP-Tasks eingesetzt werden kann. Ein solches Modell generiert für einen beliebigen Text sogenannte Embeddings, die dann weiterverarbeitet werden können. Der Vorteil an diesen ist, dass sie nicht mehr direkt Wörtern entsprechen, sondern auch den Kontext miteinbeziehen. Anhand der Embeddings kann beispielsweise unterschieden werden, worum es sich beim Wort "Bank" handelt. Das Wort besitzt unterschiedliche Bedeutungen und kann als Finanzinstitut oder als Sitzgelegenheit interpretiert werden. So können damit Modelle für die Sentiment Analysis, Text Prediction, Text Generation und andere NLP-Aufgaben trainiert werden.

Um diese Embeddings generieren zu können, muss dieser zuerst in Tokens aufgeteilt werden. Dabei resultiert aus jedem Wort ungefähr ein Token. Einzelne Worte werden jedoch in mehrere aufgeteilt, wenn sie nicht Bestandteil des Vokabular sind, das zum Trainieren verwendet wurde. Das BERT-Vokabular umfasst ca. 30'000 Wörter und Subwörter. Die Tokens werden als Zahlen dargestellt, die IDs zu den jeweiligen textlichen Teilen darstellen. Diese Zahlen stellen aber keine Reihenfolge dar und sind als kategorische Features zu betrachten.

Die Tokenliste muss jedoch immer mit dem Token *[CLS]* beginnen und mit *[SEP]* enden, da BERT damit trainiert wurde. Letzteres Token wird verwendet, um einzelne Sätze (bzw. längere Texte) voneinander zu trennen, da beim Training jeweils zwei Sätze zusammen als ein Sample verwendet wurden. Um aus den Tokens die Embeddings zu generieren, können diese als Inputs ins BERT-Modell gegeben werden, worauf dieses die Embeddings als Output liefert. Dabei reicht es, wenn der Satz bzw. kurze Text mit *[CLS]* beginnt und ein einzelnes *[SEP]* am Ende hat. Die resultierende Matrix enthält pro Token die passenden Embeddings. Um aus diesen die Sentence-Embeddings zu erhalten, können entweder die Embeddings des *[CLS]*-Tokens verwendet werden. Diese Abkürzung steht für *Classification* und fasst den Inhalt des Satzes zusammen. Eine andere Strategie nennt sich Average-Pooling, bei der der Mittelwert aller Word-Embeddings berechnet wird. Diese Sentence-Embeddings sollen den Inhalt des gesamten Texts repräsentieren. Jedes Embedding besteht aus 768 Werten, die aber nicht weiter interpretiert werden können. Diese können als Input für weitere Modelle dienen oder in unserem Fall für die Featurematrix verwendet werden.

Zu beachten ist, dass die maximale Anzahl Tokens bei 512 liegt. Längere Texte können nicht verarbeitet, beziehungsweise müssen gekürzt werden.

### Vortrainierte Modelle

Für BERT sind zwei Architekturen vorgesehen. Einmal eine einfachere, die für die meisten Anwendungen ausreicht und eine tiefere, die sogar die Fähigkeiten von Menschen bei einigen Aufgaben übertrifft. Für diese beiden Architekturen gibt es einige vortrainierte Modelle. Es empfiehlt sich diese zu verwenden, da das Training der 110 Millionen bzw. 340 Millionen Parameter einige Tage Zeit auf einem Hochleistungscomputer beansprucht. Die Basismodelle wurden mit Daten von Wikipedia und der Google Büchersammlung trainiert. Weiter gibt es spezifische Modelle für einige Sprachen. Diese vortrainierten Modelle können direkt genutzt werden, um Embeddings zu generieren oder sie können bei einem Supervised-Learning-Problem noch verbessert werden, indem sie mit eigenen Textdaten weiter trainiert werden (Transfer Learning).

Ursprünglich wurden die Modelle auf zwei Self-Supervised-Learning-Arten trainiert. Einerseits wurden bei Sätzen einzelne Wörter maskiert, die das Modell vorhersagen musste (masked language modelling). Andererseits wurden jeweils zwei Sätze aneinandergefügt, die entweder auch im ursprünglichen Text direkt aufeinanderfolgten oder nicht, und das Modell hatte die Aufgabe diese Unterscheidung zu lernen (next sentence prediction).

Für BERT gibt es einige Abwandlungen, die für spezifische Anwendungsfälle besser funktionieren sollten. So gibt es DistillBERT, dass 60 % schneller als das herkömmliche Modell sein soll, aber dennoch etwa 95 % der Qualität behalten soll. So kann das Modell auch auf weniger leistungsstarken Geräten verwendet werden.

### Transformers

Für NLP wurde bis vor einigen Jahren Modelle wie LSTM oder RNN eingesetzt, die aber insbesondere bei langen Texten zu Problemen führten. Bei letztgenannten gingen zum Beispiel vor allem Inhalte am Anfang des Textes verloren.

2017 wurde dann die Attention-Transformers-Architektur vorgestellt, die schon bald Anwendung im Bereich NLP fand. Diese Modelle versuchen auf Wichtiges zu fokussieren. Im Fall von BERT sind dies einzelne Wörter beziehungsweise Tokens. Ein weiterer Vorteil ist, dass sie durch Parallelisierung, Resultate für viel Input in viel kürzerer Zeit als die vorher genannten Architekturen liefern können.

Transformer-Modelle bestehen dabei eigentlich aus einem Encoder und einem Decoder. Ersterer maskiert, für NLP, Texte in Embeddings um. Der Decoder wandelt diese wieder in Texte oder ein anderes Output-Format um. BERT stellt jedoch nur den Encoder-Teil zur Verfügung. Die BERT-Architektur sieht im einfacheren Modell 12 Transformer-Layer und im komplexeren doppelt so viele vor.

### Referenzen

- Alammar, J. (o. J.). A Visual Guide to Using BERT for the First Time. Abgerufen 15. November 2022, von https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (arXiv:1810.04805). arXiv. https://doi.org/10.48550/arXiv.1810.04805
- Hugging Face. (o. J.-a). BERT API. Abgerufen 15. November 2022, von https://huggingface.co/docs/transformers/main/en/model_doc/bert
- Hugging Face. (o. J.-b). Bert-base-uncased. Abgerufen 15. November 2022, von https://huggingface.co/bert-base-uncased
- Muller, B. (2022, März 2). BERT 101—State Of The Art NLP Model Explained. https://huggingface.co/blog/bert-101
- Winastwan, R. (2021, November 10). Text Classification with BERT in PyTorch. Medium. https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

## Berechnen der Sentence Embeddings

Zuerst müssen die Metadaten aus der CSV-Datei eingelesen werden und Filme ohne Text im Attribut `overview` entfernt werden. Für diese können wir keine Embeddings generieren und sind deshalb für unser Recommendersystem uninteressant.
"""

# %%
meta = pd.read_csv('data/movies_meta.csv')

# %% [markdown]
"""
Glücklicherweise betrifft dies lediglich zehn Filme, die wir im nächsten Schritt aus unserem DataFrame entfernen.
"""

# %%
print('Anzahl Filme mit Wert NA in overview:', meta.overview.isna().sum())
meta = meta.dropna(subset = 'overview')
meta.shape

# %% [markdown]
"""
Nun verwenden wir unsere Library-Funktion `generateEmbeddings` um die Sentence-Embeddings zu berechnen. Diese verwendet den BERT-Tokenizer und das BERT-Modell `bert-base-uncased` aus dem Package `transformers` von [HuggingFace](https://huggingface.co/bert-base-uncased). Als Sentence-Embeddings verwenden wir die Word-Embeddings für das *[CLS]*-Token. Wir erhalten ein DataFrame zurück mit den Sentence-Embeddings pro Film, die jeweils aus 786 Elementen bestehen.
"""

# %%
n = 10

# %%
embeddings_file_cls = 'data/movies_bert_cls.csv'
sentence_embeddings = pd.read_csv(embeddings_file_cls, index_col='movieId')
# sentence_embeddings = Bertomat(BertEnum.BERT).generateEmbeddings(meta[0:n])
sentence_embeddings.shape

# %% [markdown]
"""
## Evaluierung Sentence Embeddings

Embeddings bestehen aus kontinuierlichen Werten, die sich nicht direkt interpretieren lassen. Wir wollen nun trotzdem versuchen in diesen Werten Muster zu finden. Insbesondere möchten wir diese den Genres gegenüberstellen und untersuchen, ob sich von den Embeddings auf die Genres schliessen lässt.

Aus diesem Grund lesen wir nun die Genres-Matrix ein und stellen sicher, dass beide DataFrames dieselben Filme enthalten.
"""

# %%
genres = pd.read_csv('data/movies_genres.csv', index_col='movieId')
movieIds = genres.index.intersection(sentence_embeddings.index)
sentence_embeddings = sentence_embeddings.loc[movieIds].sort_index()
genres = genres.loc[movieIds].sort_index()
genres.shape

# %% [markdown]
"""
Die Sentence-Embeddings stellen die Fimprofile dar, aus denen später in den Recommendersystemen Nutzerprofile durch Linearkombinationen gebaut werden. Diese werden dann mit der Cosine-Ähnlichkeit verglichen. Wir wollen deswegen untersuchen, wie ähnlich diese Filmprofile untereinander sind, um eine erste Einschätzung der Qualität der Recommendersysteme mit diesen Daten zu erhalten.
"""

# %%
fig, embeddings_sim = plotSimiliarities(sentence_embeddings, 'Embeddings', return_sim=True)

# %% [markdown]
"""
Es zeigt sich, dass es sehr viele ähnliche Filme gibt, denn der Grossteil der Ähnlichkeiten ist grösser als 0.9. Auch der Mittelwert liegt darüber und ist somit sehr hoch. Deswegen müssen wir später bei der Evaluierung der Recommendersysteme einen guten Grenzwert wählen, ab welcher Ähnlichkeit ein Film als gut gilt. Es können also ähnliche Filme gefunden werden, was eine wichtige Voraussetzung für die Systeme ist.

Zum Vergleich untersuchen wir die Ähnlichkeiten, wenn die Filmprofile nur aus den Genreszuweisungen bestehen würden. Wir sehen, dass es zwischen einem grossen Mehrheit der Filme keine Ähnlichkeit gibt. Da Filme mehreren Genres zugewiesen sein können, ergeben sich nicht nur die Extreme 0 und 1, sondern auch Ähnlichkeitswerte dazwischen. Der Mittelwert ist entsprechend auch viel geringer als für die Ähnlichkeitswerte der Embeddings.
"""

# %%
fig, genres_sim = plotSimiliarities(genres, 'Genres', return_sim=True)

# %% [markdown]
"""
Es gibt auch keinen linearen Zusammenhang zwischen diesen beiden Ähnlichkeiten, was der sehr kleine Korrelationskoeffizient zeigt.
"""

# %%
pearsonr(embeddings_sim.flatten(), genres_sim.flatten())

# %% [markdown]
"""
Um weiter zu prüfen, ob es einen Zusammenhang zwischen den Embeddings und den Genres gibt, wollen wir Supervised-Learning-Modelle trainieren, die aus ersterem letzeres vorhersagen können. Dabei sollen high-bias Modellarten eingesetzt werden, da diese ansonsten bei so vielen Features die Daten auch auswendig lernen könnten. Als erstes Modell setzen wir Naive-Bayes ein.
"""

# %%
nn_embeddings = sentence_embeddings - sentence_embeddings.to_numpy().min()
nb_model = MultiOutputClassifier(MultinomialNB(), n_jobs=-1)
nb_model.fit(nn_embeddings, genres)
print('Mean Accuracy:', nb_model.score(nn_embeddings, genres))

# %% [markdown]
"""
Dieses Modell schneidet ziemlich schlecht ab und kann weniger als 3 % der Filmgenres korrekt klassifizieren. Eine weitere Variante sind Support Vector Machines, die sich aber mit 768 Dimensionen nicht in vernünftiger Zeit trainieren lassen. Aus diesem Grund wollen wir eine Dimensionsreduktion mittels PCA vornehmen, aber möglichst viel Information in den Filmprofilen behalten.
"""

# %%
pca = PCA().fit(sentence_embeddings)

# %% [markdown]
"""
Die ersten Komponenten decken einen Grossteil der Varianz in den Daten ab. Mit 85 von 768 Komponenten decken wir bereits 95 % des Informationsgehalts ab.
"""

# %%
fig, n_components_bert = plotCumulativePCAVariance(pca)

# %% [markdown]
"""
Wir reduzieren nun die 768 Embeddingfeatures auf 85 und behalten so 95 % der Informationen.
"""

# %%
pca = PCA(n_components=n_components_bert)
reduced_embeddings = pca.fit_transform(sentence_embeddings)
reduced_embeddings.shape

# %% [markdown]
"""
Mit diesem kleineren Datenset können wir nun einen SVM-Classifier trainieren. Mit linearen Decision Boundaries können wir die Accuracy signifikant steigern, sie ist aber dennoch zu wenig gut als das wir von einem guten Modell sprechen könnten.
"""

# %%
svc_model = MultiOutputClassifier(SVC(kernel='linear'), n_jobs=-1)
svc_model.fit(reduced_embeddings, genres)
print('Mean Accuracy:', svc_model.score(reduced_embeddings, genres))

# %% [markdown]
"""
Ein linearer Kernel ist jedoch sehr unflexibel und es ist sehr wahrscheinlich, dass keine lineare Grenzen im PCA-Embeddings-Raum gezogen werden können. Wir ersetzen den linearen Kernel aus diesem Grund durch den flexibleren RBF-Kernel. Mit der Standardparameterisierung und einer hohen Regularisierungsstärke lässt sich kein deutlich besseres Modell trainieren. Durch Anpassung der Parametierisierung und dem Schwächen der Regularisierung, könnte das Modell zwar deutlich verbessert werden. Es wäre aufgrund der low-variance des RBF-Kernels sogar möglich ein Modell zu trainieren, dass nahezu perfekt die Genres bestimmen könnte. Dies wäre jedoch für unsere Auswertung nicht zielführend, da wir prüfen wollen, ob die Embeddings einfach auf die Genres abgebildet werden können.
"""

# %%
svc_model = MultiOutputClassifier(SVC(kernel='rbf'), n_jobs=-1)
svc_model.fit(reduced_embeddings, genres)
print('Mean Accuracy:', svc_model.score(reduced_embeddings, genres))

# %% [markdown]
"""
Als erstes Zwischenfazit können wir nun sagen, dass die Embeddings andere Informationen als die Genres abbilden und es nicht möglich ist mit einfachen Mitteln von ersterem auf letzteres zu schliessen.

### Clustering

Möglicherweise bilden sich im Embeddingsraum Gruppen von ähnlichen Filmen, die mittels Clusteringalgorithmen gefunden werden können. Diese Gruppen könnten dann ganz oder teilweise mit den Genres übereinstimmen.

Wir verwenden nun den dichtebasierten Algorithmus HDBSCAN. Im Gegensatz zum einfacheren Verfahren K-Means, detektiert dieser nicht eine fixe Anzahl an Cluster, sondern untersucht die Dichte der Datenpunkte und fasst Gruppen mit hoher Dichte zu Clustern zusammen. Cluster werden dabei hierarchisch aufgebaut und nur zusammengefasst, wenn sie genug gross sind, um sie als eigenständig zu betrachten. Datenpunkte, die zu weit entfernt von anderen sind, werden von HDBSCAN keinem Cluster zugewiesen. Dadurch könnte man diese Algorithmus auch gut zum Erkennen von Outliern verwenden.
"""

# %%
hdbscan = HDBSCAN().fit(sentence_embeddings)
print('Anzahl gefundene Cluster:', hdbscan.labels_.max() + 1)

# %% [markdown]
"""
Das Clustering hat nun drei Gruppen hervorgebracht. Wir können nun prüfen, mit welcher Wahrscheinlichkeit die einzelnen Datenpunkte zu ihrem jeweiligen Cluster gehören. Anhand des folgenden Plots sehen wir, dass die allermeisten Punkte mit sehr hoher Wahrscheinlichkeit Teil eines Clusters sind. Nur bei einzelnen konnte kein zugehöriges Cluster gefunden werden.
"""

# %%
fig = plotHDBSCANProbabilities(hdbscan)

# %% [markdown]
"""
Die cluster sind nun bestimmt und wir möchten im nächsten Schritt herausfinden, wie gut diese mit den Genres übereinstimmen. Anhand der folgenden Tabelle können wir erkennen, dass Cluster 1 mit Abstand der grösste ist und Cluster 0 und 2 jeweils nur wenige Filme enthalten. Punkte mit der Nummer -1 sind keinem Cluster zugewiesen und unabhängig voneinander.
"""

# %%
pd.Series(hdbscan.labels_).value_counts()

# %% [markdown]
"""
Weiter sehen wir im folgenden, dass Filme im Cluster 1 aus allen Genres stammen und nicht nur ein Subset abdecken. Auch bei den anderen Gruppen sowie den unabhänigen Filmen lässt sich kein Muster erkennen.
"""

# %%
fig = plotHDBSCANClustersAlongGenres(hdbscan, genres)

# %% [markdown]
"""
Dichtebasierte Algorithmen funktionieren schlechter in hochdimensionalen Räumen, was in unserem Fall mit 768 Dimensionen wahrscheinlich der Fall ist. Aus diesem Grund verwenden wir die mit PCA reduizerten Daten für ein weiteres Clustering.
"""

# %%
hdbscan_reduced = HDBSCAN().fit(reduced_embeddings)
print('Anzahl gefundene Cluster:', hdbscan_reduced.labels_.max())

# %% [markdown]
"""
Es konnte nun eine weitere Gruppe gefunden werden, aber auch hier können keine Muster erkannt werden, wenn deren Verteilungen über die verschiedenen Genres betrachtet werden.
"""

# %%
fig = plotHDBSCANClustersAlongGenres(hdbscan_reduced, genres)

# %% [markdown]
"""
Eine weitere Art der Dimensionsreduktion ist UMAP. Dabei handelt es sich im Gegensatz zu PCA um keine lineare Abbildung der Ursprungsdaten. UMAP kann aber helfen, dass die Daten auf zwei Dimensionen reduziert und somit in einem Scatterplot dargestellt werden können. Darin können dann möglicherweise Gruppen erkannt werden, die durch die Transformationen erhalten bleiben.
"""

# %%
umap = UMAP(min_dist=0.0, n_neighbors=10)
umap_embeddings = umap.fit_transform(sentence_embeddings)

# %% [markdown]
"""
Wenn wir die Datenpunkte pro Genre darstellen, können wir in den einzelnen leider keine Gruppierungen von Datenpunkten erkennen. Im Gegenteil, bei jedem Genre ist dieselbe Form zu erkennen. Es gibt abgesehen vom grossen Bogen zwar einzelne Gruppen mit wenigen Datenpunkten, aber auch diese sind bei mehreren Genres ersichtlich. Daraus kann geschlossen werden, dass die Sentence Embeddings sehr weit von den Genres entfernt sind und somit neue Informationen über die Filme liefern.
"""

# %%
fig = plotUMAPEmbeddingsAlongGenres(umap_embeddings, genres)

# %% [markdown]
"""
Es kann also nicht von den Embeddings auf die Genres geschlossen werden. Vielleicht kann anhand dieser aber das Durchschnittsrating der Filme bestimmt werden. Deshalb stellen wir in einem weiteren Plot nochmals die reduzierten UMAP-Daten dar und färben die Datenpunkte entsprechend ihrerer mittleren Bewertung ein. Aber auch hier ist kein Muster zu erkennen.
"""

# %%
ratings = pd.read_csv('movielens_data/ratings.csv')
ratings = ratings[ratings.movieId.isin(movieIds)]
mean_ratings = ratings.groupby('movieId').rating.mean().sort_index()

# %%
reduced_umap_embeddings = umap_embeddings[movieIds.isin(mean_ratings.index)]
plt.figure(figsize = (10, 5))
plt.scatter(reduced_umap_embeddings[:,0], reduced_umap_embeddings[:,1], c=mean_ratings.values, s=0.5)
plt.title('Durchschnittsrating pro Film')
plt.xlabel('UMAP Komponente 1')
plt.ylabel('UMAP Komponente 2')
plt.colorbar()
plt.show()

# %% [markdown]
"""
### Fazit

Nach diesen vielen Analysen können wir sagen, dass weder ein Zusammenhang zwischen Sentence-Embeddings und Genres noch zwischen diesen und der Durchschnittsbewertung besteht. Jedoch scheint es unter einzelnen Filmprofilen aus diesen Embeddings Zusammenhänge zu geben, was man an den Ähnlichkeitswerten und der langgezogenen Form in den mit UMAP reduzierten Daten erkennen kann. Die Filmprofile scheinen also nicht eine Sammlung von zusammenhangslosen Punkten zu sein, was für die Berechnung der Ähnlichkeiten schlecht wäre.
"""

# %% [markdown]
"""
Wir speichern die Sentence-Embeddings nun als CSV-Datei ab, damit es von der Library verwendet werden kann, um die Recommendersysteme zu bauen.
"""

# %%
sentence_embeddings.to_csv(embeddings_file_cls)

# %% [markdown]
"""
## Modellierung des Recommendersystems

Da die Featurematrix nun bereit ist, können wir diese verwenden, um unsere ersten Recommendersysteme mit diesen Daten zu bauen.
"""

# %%
matrix_cls = MatrixGenerator(bertEmbeddings=embeddings_file_cls)

# %% [markdown]
"""
Wir möchten dabei die zwei Ähnlichkeitsmasse Cosine und Pearson vergleichen. Da die Featurematrizen aus komplett anderen Daten bestehen als beim Baseline-Modell, ist es möglich, dass ein anderes Ähnlichkeitsmass hier besser funktioniert. Da mit der tertiären Ratingskala die besten Resultate beim Baseline-Recommendersystem erzielt werden konnten, werden wir diese nun auch hier einsetzen.
"""

# %%
cosine_config = CakeConfig(
    {MatrixGenerator.CONST_KEY_BERT: np.array(1)},
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE
)

pearson_config = CakeConfig(
    {MatrixGenerator.CONST_KEY_BERT: np.array(1)},
    SimilarityEnum.PEARSON,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.ZSCORE
)

cosine_eval = Evaluation('BERT_cls_cosine', cosine_config, matrix_cls)
pearson_eval = Evaluation('BERT_cls_pearson', pearson_config, matrix_cls)

mean_cosine_precision, std_cosine_precision = cosine_eval.precision()
mean_pearson_precision, std_pearson_precision = pearson_eval.precision()

cosine_topn_precision = cosine_eval.topNPrecision()
pearson_topn_precision = pearson_eval.topNPrecision()

cosine_novelty = cosine_eval.novelty()
pearson_novelty = pearson_eval.novelty()

# %% [markdown]
"""
Bei Betrachtung der Precision-Werte für 5-fache Kreuzvalidierung ist zwischen den beiden Modellen kein Unterschied zu erkennen. Mit beiden Ähnlichkeitsmassen kann ein Wert um **0.55** erreicht werden. Damit ist dieses Modell in etwa gleich gut wie unser Random-Recommendersystem.
"""

# %%
fig = plot_results(
    ['cosine', 'pearson'],
    [mean_cosine_precision, mean_pearson_precision],
    [std_cosine_precision, std_pearson_precision],
    [cosine_topn_precision, pearson_topn_precision],
    [cosine_novelty, pearson_novelty],
    'Ähnlichkeitsmassen'
)

# %% [markdown]
"""
Für diese beiden Recommendersysteme haben wir die Embeddings standardisiert. Da wir jedoch die Embeddings nicht verstehen können, können wir auch nicht beurteilen inwiefern diese Standardisierung die Embeddings beeinflusst. Als Vergleich soll nun ein Recommendersystem ohne Normalisierung erstellt werden.
"""

# %%
none_config = CakeConfig(
    {MatrixGenerator.CONST_KEY_BERT: np.array(1)},
    SimilarityEnum.COSINE,
    RatingScaleEnum.TERTIARY,
    FeatureNormalizationEnum.NONE
)

none_eval = Evaluation('BERT_cls_cosine_no-norm', none_config, matrix_cls)

mean_none_precision, std_none_precision = none_eval.precision()
none_topn_precision = none_eval.topNPrecision()
none_novelty = none_eval.novelty()

# %% [markdown]
"""
Es zeigt sich, dass mit der Standardisierung deutlich bessere Ergebnisse erzielt werden können. Das Modell ohne Standardisierung fällt hinter das Modell mit den Zufallsdaten zurück. Zwar ist die Novelty deutlich höher als mit Standardisierung, was aber nutzlos ist, wenn die Filmvorschläge den Filmschauenden nicht gefallen.
"""

# %%
fig = plot_results(
    ['standardised', 'none'],
    [mean_cosine_precision, mean_none_precision],
    [std_cosine_precision, std_none_precision],
    [cosine_topn_precision, none_topn_precision],
    [cosine_novelty, none_novelty],
    'Normalisierungen'
)

# %% [markdown]
"""
### Avg-Pooling

Bis anhin haben wir bei BERT die Embeddings des *[CLS]*-Token als Features übernommen. Eine weitere Variante um die Sentence-Embeddings zu erhalten, ist über das Mitteln der Embeddings aller Word-Tokens.
"""

# %%
embeddings_file_avg = 'data/movies_bert_avg.csv'
sentence_embeddings_avg = pd.read_csv(embeddings_file_avg, index_col='movieId')
# sentence_embeddings_avg = Bertomat(BertEnum.BERT, pooling='avg').generateEmbeddings(meta[0:n])
sentence_embeddings_avg = sentence_embeddings_avg.loc[movieIds].sort_index()
sentence_embeddings_avg.shape

# %%
sentence_embeddings_avg.to_csv(embeddings_file_avg)

# %% [markdown]
"""
Zunächst wollen wir prüfen, ob sich mit diesen Sentence-Embeddings Cluster bilden lassen, die mit den Genres übereinstimmen. Wir verwenden deswegen wieder `HDBSCAN` und erhalten dieses Mal vier Cluster.
"""

# %%
hdbscan_avg = HDBSCAN().fit(sentence_embeddings_avg)
print('Anzahl gefundene Cluster:', hdbscan_avg.labels_.max() + 1)

# %% [markdown]
"""
Die Wahrscheinlichkeiten, dass die Filme, die einem Cluster zugewiesen wurden auch zu diesem gehören ist bei den meisten bei 1, was ein gutes Zeichen ist.
"""

# %%
fig = plotHDBSCANProbabilities(hdbscan_avg)

# %% [markdown]
"""
Leider befinden sich etwa 70 % der Filme im selben Cluster und die restlichen ca. 30 % blieben ungruppiert. Die weiteren zwei Cluster enthielten lediglich zwischen 5 und 7 Datenpunkte.
"""

# %%
pd.Series(hdbscan_avg.labels_).value_counts()

# %% [markdown]
"""
Entsprechend gibt es auch hier keine Übereinstimmungen mit den Genres, wie folgender Barplot zeigt.
"""

# %%
fig = plotHDBSCANClustersAlongGenres(hdbscan_avg, genres)

# %% [markdown]
"""
Und auch mit der UMAP-Dimensionsreduktion lassen sich keine Muster auf Basis der Genres erkennen. Die Datenpunkte sind bei allen Genre-Plots über die ganze Kurve verstreut.
"""

# %%
umap_avg = UMAP(min_dist=0.0, n_neighbors=10)
umap_embeddings_avg = umap_avg.fit_transform(sentence_embeddings_avg)

# %%
fig = plotUMAPEmbeddingsAlongGenres(umap_embeddings_avg, genres)

# %% [markdown]
"""
Auch mit diesem Pooling-Verfahren erhalten wir keine Embeddings, die grosse Übereinstimmungen mit den Genres haben. Dies muss kein Problem sein, da wir durch die NLP-Modelle zusätzliche Informationen in das Modell bringen wollen.

Als nächstes wollen wir herausfinden, ob mit dem Average-Pooling ein besseres Recommendersystem erstellt werden kann. Wir verwenden erneut die tertiäre Ratingskala für die Nutzerprofile und vergleichen die zwei Ähnlichkeitsmasse Cosine und Pearson. In den Resultaten sehen wir, dass beide Masse ähnliche Werte erreichen. Die geringen Unterschiede sind nicht signifikant.
"""

# %%
matrix_avg = MatrixGenerator(bertEmbeddings=embeddings_file_avg)
cosine_avg_eval = Evaluation('BERT_avg_cosine', cosine_config, matrix_avg)
pearson_avg_eval = Evaluation('BERT_avg_pearson', pearson_config, matrix_avg)

mean_cosine_avg_precision, std_cosine_avg_precision = cosine_avg_eval.precision()
mean_pearson_avg_precision, std_pearson_avg_precision = pearson_avg_eval.precision()

cosine_avg_topn_precision = cosine_avg_eval.topNPrecision()
pearson_avg_topn_precision = pearson_avg_eval.topNPrecision()

cosine_avg_novelty = cosine_avg_eval.novelty()
pearson_avg_novelty = pearson_avg_eval.novelty()

fig = plot_results(
    ['cosine', 'pearson'],
    [mean_cosine_avg_precision, mean_pearson_avg_precision],
    [std_cosine_avg_precision, std_pearson_avg_precision],
    [cosine_avg_topn_precision, pearson_avg_topn_precision],
    [cosine_avg_novelty, pearson_avg_novelty],
    'Ähnlichkeitsmassen (avg-Pooling)'
)

# %% [markdown]
"""
## Modelle mit PCA-Dimensionsreduktion

Mit ihren 768-Dimensionen ist die Featurematrix möglicherweise zu gross und enthält zu viele Details. Mittels PCA können wir die Anzahl Dimensionen verringern und uns somit auf die grössten Unterschiede in den Daten fokussieren. Minime Unterschiede werden nicht ins reduzierte Datenset übernommen. Wir beginnen mit einer Reduktion auf nur zwei Dimensionen.
"""

# %%
pca = PCA(n_components=2)
twodim_embeddings_avg = pca.fit_transform(sentence_embeddings_avg)
twodim_embeddings_avg = pd.DataFrame(twodim_embeddings_avg, index=sentence_embeddings_avg.index)
twodim_embeddings_avg.shape

# %% [markdown]
"""
Wenn wir nun die Cosine-Ähnlichkeiten zwischen den Filmen anhand dieser zwei Features berechnen, sehen wir, dass diese einer komplett anderen Verteilung unterliegen als noch mit allen 768 Dimensionen (siehe Plot weiter oben). Wir haben nun auch Anti-Korrelationen (Werte um -1) und die Peaks befinden sich nun an beiden Enden (bei -1 und 1). In unseren Recommendersystemen verwenden wir die tatsächlichen und keine absoluten Ähnlichkeitswerte, um die Grenze zwischen guten und schlechten Übereinstimmungen zu ziehen. Eine Ähnlichkeit von -1 würde bedeuten, dass die Features der zwei Filme (bzw. des Films und Nutzprofils) komplett gegensätzlich sind und entsprechend sind die Filmschauenden nicht an solchen Vorschlägen interessiert.
"""

# %%
fig = plotSimiliarities(twodim_embeddings_avg, '2D avg-pooled Embeddings')

# %% [markdown]
"""
Wir erstellen nun einige Recommendersysteme mit unterschiedlichen Dimensionen und prüfen, mit welchem die besten Metriken erzielt werden können. Als Dimensionen werden 2, 5, 10, 100, 200 und 500 getestet.
"""

# %%
results = [{
    'dim': 768,
    'mean_precision': mean_cosine_avg_precision,
    'std_precision': std_cosine_avg_precision,
    'topn_precision': cosine_avg_topn_precision,
    'novelty': cosine_avg_novelty
}]

for dim in [2, 5, 10, 100, 200, 500]:
    pca = PCA(n_components=dim)
    reduced_embeddings_avg = pca.fit_transform(sentence_embeddings_avg)
    reduced_embeddings_avg = pd.DataFrame(reduced_embeddings_avg, index=sentence_embeddings_avg.index)

    embeddings_file_avg_reduced = f'data/movies_bert_avg_{dim}d.csv'
    reduced_embeddings_avg.to_csv(embeddings_file_avg_reduced)

    matrix = MatrixGenerator(bertEmbeddings=embeddings_file_avg_reduced)
    eval = Evaluation(f'BERT_avg_cosine_{dim}d', cosine_config, matrix)
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
Anhand der Precision können wir sagen, dass mit nur 2 Dimensionen kein gutes Recommendersystem erstellt werden kann. Dieses erreicht ähnliche Werte wie das Modell mit zufälligen Featurewerten. Mit mehr Dimensionen wird auch die allgemeine Precision und diejenige für die Top-10-Liste verbessert. Am besten funktioniert das Modell mit 100 Dimensionen. Es erreicht deutlich bessere Werte als das mit 768-Dimensionen ohne PCA-Transformation. Mit der Zunahme der Precision leidet auch die Novelty, aber bei 100 Dimensionen ist diese nur leicht niedriger als beim Höchstwert und beim kompletten Featureset.
"""

# %%
results_df = pd.DataFrame(results).sort_values('dim')
fig = plot_results(
    results_df.dim.astype(str) + 'D', results_df.mean_precision, results_df.std_precision, results_df.topn_precision, results_df.novelty, 'Dimensionen (avg-Pooling)'
)

# %% [markdown]
"""
## Fazit

Die Precision verändert sich kaum, wenn statt dem CLS-Pooling das Average-Pooling verwendet wird. Wichtig ist jedoch, dass die Features standardisiert werden. Ohne Normalisierung verschlechtern sich die Werte stark. Die Novelty verhält sich ziemlich genau gegenteilig zur Precision. Modelle mit einer hohen Precision, haben tendenziell eine niedrige Novelty.

Durch die PCA-Dimensionsreduktion auf 100 Features konnte das Recommendersystem noch ein bisschen verbessert werden, so dass es nun leicht bessere Ergebnisse als das Random-Modell liefert. Dennoch ist dieses beste BERT-Recommendersystem deutlich schlechter als das Baseline-Modell. Die Sentence-Embeddings von BERT scheinen andere Informationen ins Modell zu bringen als das Genre-Modell, die aber weniger gut für die Ähnlichkeitsberechnung geeignet sind als die Genres selbst. Möglicherweise kann eine Kombination aus Embeddings und Filmmetadaten bessere Filmvorschläge liefern.

Im folgenden Plot werden nochmals die Resultate der wichtigsten Recommendersysteme mit BERT-Embeddings präsentiert, um diese untereinander vergleichen zu können.
"""

# %%
overview_results = results_df[results_df.dim.isin([2, 100, 500])]
fig = plot_results(
    [
        'cls pooling\n no norm\n cosine',
        'cls pooling\n std\n cosine',
        'cls pooling\n std\n pearson',
        'avg pooling\n std\n cosine',
        'avg pooling\n std\n pearson',
        'PCA 2D\n avg pooling\n std\n cosine',
        'PCA 100D\n avg pooling\n std\n cosine',
        'PCA 500D\n avg pooling\n std\n cosine'
    ],
    [
        mean_none_precision,
        mean_cosine_precision,
        mean_pearson_precision,
        mean_cosine_avg_precision,
        mean_pearson_avg_precision,
        *overview_results.mean_precision
    ],
    [
        std_none_precision,
        std_cosine_precision,
        std_pearson_precision,
        std_cosine_avg_precision,
        std_pearson_avg_precision,
        *overview_results.std_precision
    ],
    [
        none_topn_precision,
        cosine_topn_precision,
        pearson_topn_precision,
        cosine_avg_topn_precision,
        pearson_avg_topn_precision,
        *overview_results.topn_precision
    ],
    [
        none_novelty,
        cosine_novelty,
        pearson_novelty,
        cosine_avg_novelty,
        pearson_avg_novelty,
        *overview_results.novelty
    ],
    'BERT-Modellen'
)

# %%
%reset -f