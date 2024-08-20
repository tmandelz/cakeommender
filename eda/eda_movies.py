# %% [markdown]
"""
# Datenanalyse für Genres und Metadaten
Dieser Teil wird verwendet, um den Datensatz der Filme sowie die dazugehörigen Metadaten zu verstehen, zu säubern und in eine Form zu bringen, mit der wir unsere Recommendersysteme aufbauen können. Metadaten, die Regisseure und Schauspieler betrifft, sind in der Datei `eda_crew_casts.py` zu finden. Ziel ist es, die Filme und ihre Metadaten als Feature-Matrix in einer CSV-Datei abzuspeichern.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect

# %%
import os
if os.getcwd().endswith('eda'):
    os.chdir('..')

# %% [markdown]
"""
## Einlesen der Daten
Die von TMDB und MovieLens zusammengeführten Metadaten werden eingelesen und angeschaut. Der Datensatz beinhaltet über 200 Features und kann im Rahmen dieses Berichts nur schlecht dargestellt werden.

"""
# %%
%%capture
movies = pd.read_csv(r"./data/movies.csv")

# %%
print('Shape des Datensatzes:', movies.shape)

# %% [markdown]
"""
Trotz der vielen Spalten möchten wir kurz die ersten fünf Zeilen zeigen, damit man ein Gefühl für den Datensatz erhält:
"""

# %%
movies.head() #some double columns like imdbId

# %%
%%capture
movies.dtypes

# %%
%%capture
print(list(movies.columns)) #column titles

# %% [markdown]
"""
## Dimensionsreduktion
### Diverse Metadaten und Spalten
Reduzieren der Metadaten, indem Produktionsländer, -firmen, Stimmen, Kollektionen, Poster, Videos, IMDB-IDs, Verweise zu Homepages und Bilder (`backdrop-path`) entfernt werden. Dies wird aufgrund unserer Projektvereinbarung so umgesetzt.
"""
# %%
movies_red = movies[movies.columns.drop(list(movies.filter(regex='production|spoken')))]

movies_red = movies_red[movies_red.columns.drop(list(movies.filter(regex='vote|collection|video|homepage|poster|imdb|backdrop')))]

# %% [markdown]

"""
### Sprachen
Wir möchten uns auf englischsprachige Filme konzentrieren, damit wir unseren NLP-Ansatz auf Basis einer Sprache umsetzen können. Danach können wir auf die Information der Sprache verzichten und löschen diese aus dem Datensatz.
Zur Verifizierung lassen wir noch eine Spracherkennungs-library über die Beschreibung des Films laufen und reduzieren den Datensatz nochmals um die Beschreibungen, die nicht als Englisch erkannt wurden.

<br>
<br>
Wir verwenden die Python-Library `langdetect` für diesen Task.
Sie funktioniert mit einem Naive-Bayesian-Filter, welcher die Wahrscheinlichkeit berechnet, dass ein Text in einer gewissen Sprache geschrieben ist.
Dafür wurden auf Basis von Wikipedia-Artikeln Sprachprofile in über 50 Sprachen definiert.
Die Precision der Erkennung liegt bei 99 % für die definierten Sprachen.
Es können auch mehrere Sprachen erkannt werden, dann gibt die Library mehrere Wahrscheinlichkeiten aus.
<br>
<br>
14 weitere Filme konnten so nochmals entfernt werden deren Beschreibung nicht in Englisch waren.
"""

# %%
movies_red = movies_red.drop(movies_red[movies_red.original_language != 'en'].index)
movies_red = movies_red.drop(['original_language'], axis=1)

# %%
def detect_language(text:str) -> str:
    """
    Daniela Herzig
    Function to detect language in the column "overview"
    :param text: string in the column overview
    :return string of detected language
    """
    try:
        return detect(text)
    except:
        return 'unknown'

movies_red['language'] = movies_red['overview'].apply(detect_language)

# %%
movies_red['language'].value_counts() # 14 not as english detected overviews
movies_red.drop(movies_red[movies_red.language != 'en'].index, inplace=True)
movies_red = movies_red.drop(['language'], axis=1)

# %% [markdown]
"""
### IDs
Es gibt verschiedene IDs in unserem Datensatz, die höchst wahrscheinlich doppelt vorhanden sind. Wir prüfen ob die Hypothese stimmt, dass die Spalten `tmbdId` und `id` die gleichen Werte enthalten.
<br>
Da dies tatsächlich der Fall ist, entfernen wir die Spalte `id`.
"""

# %%
%%capture
movies_red[movies_red['id'] != movies_red['tmdbId']] #same id

# %%
movies_red = movies_red.drop(['id'], axis=1)

# %% [markdown]
"""
### Status der Filme
Wir untersuchen nun den Status der Filme. Wir entfernen die geplanten oder noch in Produktion befindlichen Filme sowie aller weiteren Spalten, die Informationen zum Status aufzeigen (wie `status_code`, `status_message` und `success`).
"""
# %%
movies_red['status'].value_counts()

# %%
movies_red = movies_red.drop(["status"], axis=1)
movies_red = movies_red.drop(["status_code"], axis=1)
movies_red = movies_red.drop(["status_message"], axis=1)
movies_red = movies_red.drop(["success"], axis=1)

# %% [markdown]
"""
### Adult
Die Spalte `adult` liefert uns keine bedeutende Informationen, da es nur zwei Werte beinhaltet. Wir verzichten auf die zwei Filme, die bei adult `true` angeben und entfernen sie.
"""

# %%
#only two true values -> ignoring this feature
movies_red['adult'].value_counts()

# %%
movies_red = movies_red.drop(['adult'], axis=1)

# %% [markdown]
"""
### Titel
Wir haben diverse Spalten für den Filmtitel. Die Spalte `title` kommt von TMDB, `title.1` ist aus dem MovieLens-Datensatz und beinhaltet zusätzlich das Erscheinungsjahr in Klammern, und `original_title` ist der Originaltitel. Wir werden den Original-Titel weiterverwenden und entfernen die anderen Titel.
"""

# %%
movies_red = movies_red.drop(['title', 'title.1'], axis=1)

# %% [markdown]
"""
### Erscheinungsdatum
Alle Filme, die vor 2010 erschienen sind und alle, die nach dem Herunterladen des Datensatzes erscheinen sollen, werden aus dem Datensatz gelöscht. Dies ist eine Verifizierung der bereits bei der Datenbeschaffung berücksichtigen Erscheinungsdaten, die aber dieses Mal auf den TMDB-Metadaten beruht.
"""

# %%
movies_red['release_date'] = pd.to_datetime(movies_red['release_date'])
#movies_red.dtypes

# %%
movies_red.drop(movies_red[(movies_red['release_date'] < '2010-01-01') & (movies_red['release_date'] > '2022-10-16')].index, inplace=True)

# %% [markdown]
"""
### Tagline
Die Tagline, die einen Kurz-Teaser für den Film darstellt, werden wir für unsere NLP-Ansätze nicht benötigen, daher entfernen wir diese aus dem Datensatz.
"""

# %%
movies_red.drop(['tagline'], axis=1, inplace=True)

# %% [markdown]
"""
### Popularity
Die Popularität ist abhängig von der Anzahl Votes und den als Favoriten gespeicherter Filme von TMDB-Nutzern. Da dies sehr tagesabhängig ist und dadurch zur Ähnlichkeitsberechnung eher ungeeignet ist, verzichten wir auf diese Information.
"""

# %%
movies_red.drop(['popularity'], axis=1, inplace=True)

# %% [markdown]
"""
### Genres
Wir besitzen Informationen über die Genres sowohl von TMDB als auch von MovieLens. Diese sollten zu einem grossen Teil übereinstimmen. Die Genre-IDs benötigen wir nicht und werden aus dem Datensatz gelöscht.
"""

# %%
movies_red = movies_red[movies_red.columns.drop(list(movies.filter(regex=r"genres\d_id")))]

# %% [markdown]
"""
#### TMDB-Genres
Wir extrahieren die TMDB-Genres und untersuchen diese.
"""

# %%
genres = movies_red.filter(regex=r"genres\d_name")

# %% [markdown]
"""
Es gibt einige unterschiedliche Genres bei TMDB, wie der folgende Plot zeigt. Der Plot gibt uns keine Information über die gesamte Verteilung aller Genres, sondern zeigt uns nur beispielhaft, wie diese auftreten.
"""

# %%
plt.figure(figsize=(10, 9))
genres['genres0_name'].value_counts().plot(kind='barh')
plt.xlabel("Anzahl Filme")
plt.ylabel("Genres von TMDB (nur erstgenanntes)")
plt.title("Verteilung des erstgenannten Genres bei TMDB")
plt.figtext(0.8, -0.01, f"n = {len(genres.genres0_name)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
Im nächsten Plot sehen wir, dass es nur beim erstgenannten Genre kaum NAs gibt. Weiter zeigt die untere Grafik, dass fast keinen Filmen mehr als vier Genres zugewiesen wurden.
"""

# %%
plt.figure(figsize=(10, 4))
genres.isna().sum().plot(kind='barh')
plt.xlabel("Anzahl NAs bei TMDB-Genres")
plt.ylabel("Genre-Spaltenindex")
plt.title("NAs in TMDB-Genres")
plt.figtext(0.8, -0.01, f"n = {len(genres)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
#### MovieLens-Genres
"""

# %%
#movielens genres
genresML = movies_red['genres'].str.split(pat='|', expand=True)

# %% [markdown]
"""
Im Plot sehen wir, dass bei über 600 Filmen keine Informationen zu Genres im MovieLens-Datensatz enthalten sind. Wir untersuchen im folgenden die NAs sowie fehlende Informationen in den Genres-DataFrames.
"""

# %%
plt.figure(figsize=(10, 7))
genresML[0].value_counts().plot(kind='barh')
plt.xlabel("Anzahl Filme")
plt.ylabel("Genres von MovieLens (nur erstgenanntes")
plt.title("Verteilung des erstgenannten Genres bei MovieLens")
plt.figtext(0.8, -0.01, f"n = {len(genresML[0])}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
Im Plot sehen wir, dass es nur beim erstgenannten Genre keine NAs (aber dafür `no genres listed`) gibt. Bei genauerer Untersuchung des Datensatz wird erkannt, dass es nur in der ersten Genre-Spalte `no genres listed` erwähnt wird. In den weiteren Spalten gibt es stattdessen NAs. Weiter zeigt die untere Grafik, dass fast alle Filme höchstens drei Genres haben.
"""

# %%
plt.figure(figsize=(10, 5))
genresML.isna().sum().plot(kind='barh')
plt.xlabel("Anzahl NAs bei MovieLens-Genres")
plt.ylabel("Genre-Spaltenindex")
plt.title("NAs in MovieLens-Genres")
plt.figtext(0.8, -0.01, f"n = {len(genresML)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
#### Fazit für Genres

Die Kategorie `no genres listed` kommt nur in der ersten Spalte des Datensatzes von MovieLens vor, heisst also, alle weiteren folgenden sind dann als NAs im Datensatz. Aufgrund der doch sehr wenigen Genres und den 636 Filmen ohne Genres, werden wir uns auf die Genres-Einteilung von TMDB konzentrieren und die MovieLens-Daten im Datensatz löschen. Wir möchten alle Genres der TMDB-Datenbank behalten, da wir nicht wissen, in welcher Reihenfolge die Genres aufgelistet sind und wir keine wichtigen Daten verlieren möchten.
"""

# %%
%%capture
genresML[0].value_counts()['(no genres listed)']

# %%
%%capture
genresML[1].value_counts()

# %% [markdown]
"""
#### Überlappung der Genres von TMDB und MovieLens

Wir möchten trotzdem noch untersuchen, ob es eine gewisse Übereinstimmung zwischen den Genres von TMDB und MovieLens gibt. Hierzu werden zuerst alle Genres von beiden Datensätzen bestimmt.
"""

# %%
genresTmdb = pd.unique(genres.values.ravel('K'))

# %%
genresMovieL = pd.unique(genresML.values.ravel('K'))

# %% [markdown]
"""
Genres-Kategorien, die nicht in beiden Datensätzen vorkommen, sind hier aufgelistet:
"""

# %%
genresTmdb[~np.isin(genresTmdb, genresMovieL)]
# %%
genresMovieL[~np.isin(genresMovieL, genresTmdb)]

# %% [markdown]
"""
Wir erkennen, dass es einige Genres gibt, die das gleiche oder ähnliche meinen, aber anders kategorisiert sind. So werden `Sci-Fi` und `Science Fiction` gleich benannt sowie `Family` und `Children`, weswegen wir diese entsprechend umkodieren.
"""

# %%
genresML.replace('Sci-Fi', 'Science Fiction', inplace=True)
genresML.replace('Children', 'Family', inplace=True)

# %%
genres = pd.concat([genres, genresML], axis=1)

# %%
genresValue = genres.apply(pd.value_counts)

# %%
genresValue['TMDB'] = genresValue.loc[:, 'genres0_name':'genres7_name'].sum(axis=1)
genresValue['MovieLens'] = genresValue.iloc[:, 8:18].sum(axis=1)

# %% [markdown]
"""
In der folgenden Grafik ist erkennbar, dass es filmunabhängig eine grosse Überschneidung der Genres zwischen den zwei Datensets gibt. Um eine genaue Aussage hierzu zu machen, müsste dies noch filmweise untersucht werden. Da wir uns aber entschieden haben, uns nur auf einen Genres-Datensatz - den TMDB-Datensatz - zu fokussieren, werden wir dies nicht mehr vertiefen.
"""

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
genresValue[['TMDB', 'MovieLens']].plot(kind='bar', ax=ax)
plt.xlabel("Genres")
plt.ylabel("Anzahl Filme")
plt.title(f"Vergleich der Genres zwischen den Datensätzen")
plt.show()

# %%
movies_red.drop(['genres'], axis=1, inplace=True)

# %% [markdown]
"""
## Prüfen des reduzierten Datensatzes
Der Datensatz ist nun reduziert und beinhaltet nur noch die Informationen, welche wir für unsere Recommendersysteme verwenden möchten. Nun werden diese bezüglich ihrer Werte überprüft.

### NAs
Voraussetzung ist, dass wir jeden Film identifizieren können, daher benötigen wir die movieId und tmdbId. Alle anderen Werte dürfen NAs enthalten.
"""

# %%
print('Anzahl fehlende Werte in movieId:', movies_red['movieId'].isna().sum())
print('Anzahl fehlende Werte in tmdbId: ', movies_red['tmdbId'].isna().sum())

# %% [markdown]
"""
### Budget-Verteilung

Wir haben sehr viele 0-Werte. Diese scheinen unplausibel und damit diese nicht das Recommendersystem verfälschen, werden diese durch NA-Werte ersetzt.
"""

# %%
movies_red['budget'].describe()

# %% [markdown]
"""
Man kann in der folgenden Boxplotgrafik erkennen, dass es einige Ausreisser bei hohen Budgets gibt.
"""

# %%
movies_red['budget'].replace(0, np.nan, inplace=True)
plt.figure(figsize=(10, 10))
sns.boxplot(x=movies_red['budget'] / 1e6, y=movies_red['genres0_name'])
plt.xlabel("Budget [M$]")
plt.ylabel("erstgenanntes Genre")
plt.title("Filmbudgets")
plt.show()

# %% [markdown]
"""
Weiter ist in der nächsten Grafik erkennbar, dass es sehr viele kleine Budgets gibt. Ob dies sinnvolle Werte sind, ist uns nicht ganz klar. Wir behalten sie deswegen im Datenset.
"""

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.histplot(x=movies_red['budget'] / 1e6, ax=ax)
plt.xlabel("Budget [M$]")
plt.ylabel("Anzahl Filme")
plt.title(f"Filmbudgets")
plt.show()

# %% [markdown]
"""
### Einnahmen
Auch hier haben wir sehr viele 0-Werte, die wir auch durch NAs ersetzen, um das Recommendersystem nicht zu verfälschen.
"""

# %%
movies_red['revenue'].describe()

# %%
movies_red['revenue'].replace(0, np.nan, inplace=True)

# %% [markdown]
"""
Es wäre spannend zu untersuchen, ob das Budget eines Films mit den Einnahmen korreliert. Wir haben diesen Zusammenhang in der folgenden Grafik geplottet. Die meisten Filme haben Einnahmen von bis zu 350 Millionen Dollar und es gibt einzelne Filme, die sehr hohe Einnahmen haben (knapp 2 Milliarden). Wir mussten dies mit einer Google-Suche verifizieren: Es gibt tatsächlich so hohe Einnahmen.
"""

# %%
plt.figure(figsize=(10, 7))
sns.scatterplot(x=movies_red['revenue'] / 1e6, y=movies_red['budget'] / 1e6)
plt.xlabel("Umsatz [M$]")
plt.ylabel("Budget [M$]")
plt.title("Zusammenhang von Umsatz und Budget")
plt.show()

# %% [markdown]
"""
### Filmlänge
Wir haben hier auch einige 0-Werte, aber nicht so viele wie bei den anderen Attributen. Auch hier werden diese durch NA-Werte ersetzt.
Die meisten Filme haben eine Länge von 80 bis 120 Minuten, also klassische Kino-Filmlängen, wie wir sie kennen.
"""

# %%
movies_red['runtime'].describe()

# %% [markdown]
"""
Die These, dass lange Filme auch ein höheres Budget benötigen, kann eindeutig widerlegt werden mit der folgenden Grafik.
"""

# %%
plt.figure(figsize=(10, 6))
movies_red['runtime'].replace(0, np.nan, inplace=True)
sns.scatterplot(x=movies_red['runtime'], y=movies_red['budget'] / 1e6)
plt.xlabel("Dauer [min]")
plt.ylabel("Budget [M$]")
plt.title(f"Zusammenhang zwischen Dauer und Budget")
plt.show()

# %%
%%capture
sns.displot(x=movies_red['runtime'])
plt.xlabel("runtime [min]")
plt.ylabel("count of movies")
plt.title(f"runtime of movies")
plt.show()

# %% [markdown]
"""
## Feature-Matrix erstellen
Zum Schluss erstellen wir die Feature-Matrix für unsere Recommendersysteme und exportieren diese als CSV-Datei für die weitere Nutzung im Rahmen dieser Challenge. Dabei werden Dummies für die Genres erstellt. Für die Metadaten benötigt es dies nicht, da diese hauptsächlich aus kontinuierlichen besteht und keine kategorischen Variablen enthält.
"""

# %%
feature = movies_red[['movieId', 'genres0_name', 'genres1_name', 'genres2_name', 'genres3_name', 'genres4_name', 'genres5_name', 'genres6_name', 'genres7_name']]
# %%
feature = pd.melt(feature, id_vars=['movieId'], value_vars=['genres0_name', 'genres1_name', 'genres2_name', 'genres3_name', 'genres4_name', 'genres5_name', 'genres6_name', 'genres7_name' ], var_name='feature').drop('feature', axis=1).dropna()
# %%
feature = pd.get_dummies(feature, prefix="", prefix_sep="").groupby('movieId').sum()
# %%
%%capture
feature.iloc[:,1:19].isin([0,1]).all()
# %%
feature.to_csv("data/movies_genres.csv")
# %%
movies_red = movies_red[(movies_red.columns.drop(list(movies_red.filter(regex='genres'))))]

# %%
movies_red.reset_index(drop=True, inplace=True)
movies_red = movies_red.set_index('movieId')
# %%
movies_red.to_csv(r"./data/movies_meta.csv")


