# -*- coding: utf-8 -*-
# ---
# jupyter:
#   author: 'Daniela Herzig, Jan Zwicky, Joseph Weibel, Thomas Mandelz'
#   title: 'Report 2 - Datenanalyse für Schauspieler und Regisseure'
# ---

# %% [markdown]
"""
# Datenanalyse für Schauspieler und Regisseure
Dieser Teil wird verwendet zur explorativen Datenanalyse, Dimensionsreduktion und Dummyerstellung der Schauspieler und Regisseure.
Insbesondere soll herausgefunden werden wieviele Einzelschauspieler und Regisseure wir in unserem Datensatz haben und wie die Verteilung der Auftritte dieser ist.
"""

#%%
import os
if os.getcwd().endswith('eda'):
    os.chdir('..')

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %% [markdown]
"""
## Einlesen der Schauspieler und Crew Metadaten
Die von TMDB heruntergeladenen Metadaten werden in ein DataFrame eingelesen.
"""

# %%
%%capture
dfMoviesCastsCrew = pd.read_csv(r"./data/movies_casts_crew.csv")

# %% [markdown]
"""
## Dimensionsreduktion durch Spaltenentfernung
Da wir sämtliche Informationen heruntergeladen haben, gibt es Spalten, die wir nicht benötigen.
Diese Spalten werden mittels Regex extrahiert und anschliessend aus dem DataFrame entfernt.
Wir möchten nur die folgenden Spalten behalten:

| Spaltename              	| Spaltenbeschreibung             	|
|---|---|
| movieId | ID von MovieLens |
| tmdbId | ID von TMDB |
| title | Filmtitel |
| cast{index}_name | Namensspalte eines Schauspielers/einer Schauspielerin |
| crew{index}_name | Namensspalte eines Regisseurs/einer Regisseurin |

"""

# %%
import warnings
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression, and has match groups.')
removeColumns = dfMoviesCastsCrew.columns[~dfMoviesCastsCrew.columns.str.contains(
    "(cast\d{1,2}_name)|crew0_name|(^[_]id)|movieId|title|tmdbId", regex=True)]
dfMoviesCastsCrewTruncated = dfMoviesCastsCrew[dfMoviesCastsCrew.columns.drop(removeColumns)]

# %% [markdown]
"""
## Zwischenspeichern
Das Resultat der Dimensionsreduktion wird nun zwischengespeichert und als neue Grundlage für die Weiterverarbeitung verwendet.
"""

# %%
dfMoviesCastsCrewTruncated.to_csv(r"./data/movies_crew_casts_truncated.csv")

# %% [markdown]
"""
## Datenvalidierung
Als erstes validieren wir die Dimensionsreduktion indem wir die ersten Zeilen begutachten.
"""

# %%
dfMoviesCastsCrewTruncated.head()

# %% [markdown]
"""
Die Reduktion hat gut funktioniert. Wir haben eine Reduktion von ca 3500 Spalten:
"""

# %%
print('Anzahl entfernter Spalten:', len(dfMoviesCastsCrew.columns) - len(dfMoviesCastsCrewTruncated.columns))

# %% [markdown]
"""
Bevor wir weiter mit den Daten arbeiten, prüfen wir zuerst auf etwaige NA-Werte.
Im Cast erwarten wir NA-Werte, da nicht alle Filme gleich viele Auftritte haben.
Die Spalten `movieId`, `tmdbId` und `title` dürfen keine NAs enthalten.
"""

# %%
print('NAs in movieId:', sum(dfMoviesCastsCrewTruncated["movieId"].isna()))
print('NAs in tmdbId:', sum(dfMoviesCastsCrewTruncated["tmdbId"].isna()))
print('NAs in title:', sum(dfMoviesCastsCrewTruncated["title"].isna()))

# %% [markdown]
"""
Die Spalten haben tatsächlich keine NA-Werte. Wir prüfen nun die Verteilung der NA-Werte in den restlichen Spalten.
Wir plotten diese als Histogram der Anzahl NA-Werte.
"""

# %%
naValuesColumns = dfMoviesCastsCrewTruncated.loc[:, ~dfMoviesCastsCrewTruncated.columns.str.contains(
    "movieId|tmdbId|title")].isna().sum()
plt.figure(figsize=(10, 5))
plt.hist(naValuesColumns, bins=50)
plt.xlabel("Anzahl NA-Werte")
plt.ylabel("Anzahl Spalten")
plt.title(f"Verteilung der Anzahl NA-Werte über die Spalten")
plt.figtext(0.8, -0.01, f"n={len(naValuesColumns)} "f"\n Total Anzahl Spalten= {len(dfMoviesCastsCrewTruncated)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
Viele der Spalten enthalten praktisch überall NAs. Da wir bis zu 101 Spalten mit Schauspieler\*innen und Regisseur\*innen pro Film haben, ist dies keine Überraschung.
Es gibt auch kaum Filme, die bis zu 101 Besetzungen haben.

Wir ziehen daraus das Fazit, dass keine NA Bereinigungen oder Imputationen erfolgen müssen.
"""

# %% [markdown]
"""
## Verteilung Auftritte der Schauspieler\*innen
Bis jetzt wissen wir noch nicht wie viele Einzelschauspieler\*innen wir in unserem Datenset haben. Auch deren Verteilung der Auftritte wissen wir nicht.
Um einen Überblick über die Dimensionen zu bekommen, werden wir zuerst die Einzelschauspieler\*innen und deren Auftritte quantifizieren.
Wir erstellen dazu ein DataFrame welches nur Variablen des Cast sowie die `movieId` beinhaltet.
"""

# %%
%%capture
removeColumnsCasts = dfMoviesCastsCrewTruncated.columns[~dfMoviesCastsCrewTruncated.columns.str.contains(
    "cast|movieId", regex=True)]
dfMoviesCastsTruncated = dfMoviesCastsCrewTruncated[dfMoviesCastsCrewTruncated.columns.drop(
    removeColumnsCasts)]

dfMoviesCastsTruncated.head()

# %% [markdown]
"""
Nun können wir aus diesem Subset mittels der pandas `melt` Funktion das Datenset lang machen.
Auf dieses lange Datenset wenden wir die Funktion `value_counts` an.
Das Resultat ist eine Liste der Einzelschauspieler\*innen und die Gesamtanzahl ihrer Auftritte.
"""

# %%
castCounts = dfMoviesCastsTruncated.melt()["value"].value_counts()
castCounts

# %% [markdown]
"""
Die Einzelschauspieler\*innen sind nun nach Anzahl Auftritten geordnet. Um einen besseren Überblick über die Verteilung zu erhalten, schauen wir uns zuerst einige Kennzahlen an.
"""

# %%

describeStats = castCounts.describe()
describeStats.at["median"] = castCounts.median()
describeStats

# %% [markdown]
"""
Folgendes fällt uns auf:

- Der Minimalwert ist bei einem Auftritt. Dies ergibt Sinn da wir keine Schauspieler\*innen wollen die nicht mindestens in einm Film vorkommen.
- Der Maximalwert ist bei 64 Auftritten.
- Der Mittelwert ist bei ca 1.7 Auftritten. Dies ist sehr tief.
- Der Median ist bei einem Auftritt. Dies ist ebenfalls sehr tief und bedeutet, dass mehr als die Hälfte der Schauspieler\*innen lediglich einen Auftritt hatten.
- Das 3. Quartil ist bereits bei zwei Auftritten. Mehr als 3/4 der Schauspieler haben somit nur 1-2 Auftritte. Diese Schauspieler sind für das Recommendersystem vermutlich nicht sehr hilfreich.
<br>
<br>
"""

# %% [markdown]
"""
Als nächstes schauen wir uns die Verteilung der Auftritte genauer an.
"""

# %%
plt.figure(figsize=(10, 4))
plt.hist(castCounts.values, bins=50)
plt.xlabel("Anzahl Auftritte")
plt.ylabel("Anzahl Schauspieler*innen")
plt.title("Verteilung Auftritte in Filmen")
plt.figtext(0.8, -0.01, f"n={len(castCounts)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
In der Verteilung ist klar erkennbar, dass die meisten Schauspieler\*innen zwischen einem und 5 Auftritte haben.
Die oberen Ausreisser sind gar nicht mehr erkennbar.
"""

# %% [markdown]
"""
## Dimensionsreduktion Schauspieler\*innen
Da die Anzahl Einzelschauspieler\*innen einen Einfluss auf die Dimension der Featurematrix im Recommendersystem hat, müssen wir auch bei den Einzelschauspielern\*innen eine Reduktion vornehmen.
"""

# %% [markdown]
"""
Unsere erste Annahme war, dass Einzelschauspielern\*innen, welche nicht mindestens 10 Auftritte haben, für das Recommendersystem nicht relevant sind, da es schwierig wird anhand dieser ähnliche Filme zu bestimmen.
"""

# %%
castCountsCut = castCounts[castCounts >= 10]
plt.figure(figsize=(10, 4))
plt.hist(castCountsCut.values, bins=50)
plt.xlabel("Anzahl Auftritte")
plt.ylabel("Anzahl Schauspieler*innen")
plt.title("Verteilung Auftritte in Filmen (mind. 10 Auftritte)")
plt.figtext(0.8, -0.01, f"n={len(castCountsCut)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
Diese Annahme führte zu einer starken Reduktion des Casts.
Jedoch sind wir immernoch bei über 2'000 Schauspieler\*innen.

<br>
Grundsätzlich ist diese Schwelle ein Hyperparameter, der mittels Kreuzvalidierung über verschiedene Thresholds evaluiert werden muss, für welchen jeweils ein neues Recommendersystem erstellt werden muss.
Da wir vermutlich nicht die Rechenleistung und Zeit haben um diesen Threshold sehr hoch anzusetzen, haben wir uns für den trivialen Ansatz entschieden einige gängige Quantile (Top 20 %, 10 %, 5 %, 1 %) auszuprobieren bis die Menge von Schauspieler\*innen unter 2'000 ist.
"""

# %%
castCountsCut = castCounts[castCounts > castCounts.quantile(.99)]
plt.figure(figsize=(10, 6))
plt.hist(castCountsCut.values, bins=50)
plt.xlabel("Anzahl Auftritte")
plt.ylabel("Anzahl Schauspieler*innen")
plt.title("Verteilung Auftritte in Filmen (Top 1 %)")
plt.figtext(0.8, -0.01, f"n={len(castCountsCut)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
Mit den Top 1 % der Schauspieler\*innen sind wir bei ca. 1'500 Einzelschauspieler\*innen.
<br>
Wir behalten uns vor diese Schwelle im Verlauf der Challenge nochmals anzupassen.
"""

# %% [markdown]
"""
## Verteilung Produktionen der Regisseur\*innen
Bis jetzt wissen wir noch nicht wie viele Einzelregisseur\*innen wir in unserem Datenset haben. Auch deren Verteilung der Produktionen wissen wir nicht.
Um einen Überblick über die Dimensionen zu bekommen, werden wir zuerst die Einzelregisseur\*innen und deren Produktionen quantifizieren.
Wir erstellen dazu ein DataFrame, das nur Variablen der Crew sowie die `movieId` beinhaltet.
"""

# %%
removeColumnsCrew = dfMoviesCastsCrewTruncated.columns[~dfMoviesCastsCrewTruncated.columns.str.contains(
    "crew|movieId", regex=True)]
dfMoviesCrewsTruncated = dfMoviesCastsCrewTruncated[dfMoviesCastsCrewTruncated.columns.drop(
    removeColumnsCrew)]
dfMoviesCrewsTruncated

# %% [markdown]
"""
Auch hier bringen wir das Subset in ein langes Format. So können wir die Anzahl Produktionen pro Regisseur\*in bestimmen.
"""

# %%
crewCounts = dfMoviesCrewsTruncated.melt()["value"].value_counts()
crewCounts

# %% [markdown]
"""
Die Einzelregisseur\*innen sind nun nach Anzahl Produktionen geordnet. Um einen besseren Überblick über die Verteilung zu erhalten, schauen wir uns zuerst einige Kennzahlen an.
"""

# %%
describeStats = crewCounts.describe()
describeStats.at["median"] = crewCounts.median()
describeStats

# %% [markdown]
"""
Anhand dieser Kennzahlen können wir folgendes bestimmen:
- Der Minimalwert ist bei einem Auftritt. Dies ergibt Sinn da wir keine Regisseur\*innen wollen, die nicht mindestens in einem Film vorkommen.
- Der Maximalwert ist bei 27 Produktionen.
- Der Mittelwert ist bei ca. 1.2 Produktionen, was sehr tief ist.
- Der Median ist bei einer Produktion.
- Das 3. Quartil ist immer noch bei einer Produktion. Mehr als 3/4 der Regisseur\*innen haben somit nur bei einer Produktion mitgearbeitet.
Diese Regisseur\*innen sind für das Recommendersystem vermutlich nicht sehr hilfreich.
"""

# %% [markdown]
"""
Als nächstes betrachten wir die Verteilung der Regisseur\*innen genauer.
"""

# %%
plt.figure(figsize=(10, 4))
plt.hist(crewCounts.values, bins=50)
plt.xlabel("Anzahl Produktionen")
plt.ylabel("Anzahl Regisseur*innen")
plt.title("Verteilung Produktionen der Regisseur*innen")
plt.figtext(0.8, -0.01, f"n={len(crewCounts)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
In der Verteilung ist klar erkennbar, dass die meisten der Regisseur\*innen zwischen 1-2 Produktionen haben.
Die oberen Ausreisser sind gar nicht mehr erkennbar.
"""

# %% [markdown]
"""
## Dimensionsreduktion Regisseur\*innen
Da die Anzahl Einzelregisseur\*innen einen Einfluss auf die Dimension der Featurematrix im Recommendersystem hat, müssen wir auch bei den Einzelregisseur\*innen eine Reduktion vornehmen.
"""

# %% [markdown]
"""
Unsere erste Annahme war, dass Einzelregisseur\*innen, die nicht mindestens 5 Produktionen haben, für das Recommendersystem nicht relevant sind.
"""

# %%
crewCountsCut = crewCounts[crewCounts >= 5]
plt.figure(figsize=(10, 4))
plt.hist(crewCountsCut.values, bins=50)
plt.xlabel("Anzahl Produktionen")
plt.ylabel("Anzahl Regisseur*innen")
plt.title("Verteilung Produktionen der Regisseur*innen (mind. 5 Produktionen)")
plt.figtext(0.8, -0.01, f"n={len(crewCountsCut)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
Diese Annahme führte zu einer starken Reduktion der Einzelregisseur\*innen.

<br>
Auch hier kann mittels Hyperparametertuning die optimale Anzahl Einzelregisseur\*innen bestimmt werden. Wir wollen aber auch hier den trivialen Ansatz verwenden und einige gängige Quantile (Top 20 %, 10 %, 5 %, 1 %) ausprobieren bis die Menge von Einzelregisseur\*innen unter 500 ist.
"""

# %%
crewCountsCut = crewCounts[crewCounts > crewCounts.quantile(.99)]
plt.figure(figsize=(10, 4))
plt.hist(crewCountsCut.values, bins=50)
plt.xlabel("Anzahl Produktionen")
plt.ylabel("Anzahl Regisseur*innen")
plt.title("Verteilung Produktionen der Regisseur*innen (Top 1 %)")
plt.figtext(0.8, -0.01, f"n={len(crewCountsCut)}",
            wrap=True, horizontalalignment='center', fontsize=8)
plt.show()

# %% [markdown]
"""
Mit den Top 1 % der Regisseure sind wir bei ca. 260 Einzelregisseur\*innen.
Wir halten uns vor diese Schwelle im Verlauf der Challenge nochmals anzupassen.
"""

# %% [markdown]
"""
## Dummies Featurematrix Schauspieler & Regisseure

Die Dimensionsreduktion und die explorative Datenanalyse waren erst die Vorbereitung des Endprodukts.
Für das Recommendersystem müssen wir eine Featurematrix erstellen, die als Spalte die Namen der Schauspieler\*innen und Regisseur\*innen besitzt und als Index die Movie-ID.
Sofern ein*e Schauspieler\*in oder Regisseur\*in im Film vorkommt, soll in der Zelle der Wert 1 stehen, ansonsten 0.
"""

# %% [markdown]
"""
Hierzu sammeln wir zuerst alle möglichen Spaltenamen (`cast\d{1,2}_name`, `crew0_name`) in je einer Liste.
"""

# %%
%%capture
castColumns = dfMoviesCastsCrew.columns[dfMoviesCastsCrew.columns.str.contains(
    "(cast\d{1,2}_name)", regex=True)]
castColumns

# %%
%%capture
crewColumns = ['crew0_name']
crewColumns

# %% [markdown]
"""
Nun sammeln wir alle Namen von Schauspielern und Regisseuren aus den Reduktionen, die wir bereits erstellt haben.
"""

# %%
%%capture
removeListCasts = list(castCountsCut.index)
removeListCasts

# %%
%%capture
removeListCrews = list(crewCountsCut.index)
removeListCrews

# %% [markdown]
"""
Mittels den oben erstellten Listen und der selbsterarbeiteten Funktion `createFeatureMatrix` kann eine Dummy-Feature-Matrix für Schauspieler\*innen und Regisseur\*innen erstellt werden.
"""

#%%
def createFeatureMatrix(csvSavePath: str, dfMoviesTruncated: pd.DataFrame, meltColumns: pd.core.indexes.base.Index, removeList: list, indexId: str = "movieId") -> pd.DataFrame:
    """
        Thomas Mandelz
        Function to create a featureMatrix from a truncated Movies Dataframe and Columns to Melt
        :param csvSavePath: file path to save the csv : str
        :param dfMoviesTruncated: Dataframe with Truncated Movies : pd.Dataframe
        :param meltColumns: List of Columns to melt to create a long df: pd.core.indexes.base.Index
        :param removeList: list of column names to remove after melt : list of strings
        :param indexId: string of index name to keep : str
        :return: pd.Dataframe as a Featurematrix
        :rtype: pd.Dataframe
    """
    # melt from wide to long, drop the "Feature" variable, drop na (NA's were cast columns which had no entry)
    dfMoviesLong = dfMoviesTruncated.melt(
        indexId, value_vars=meltColumns, var_name="Feature").drop("Feature", axis=1).dropna()
    # Remove casts which are cut
    dfMoviesCut = dfMoviesLong[dfMoviesLong['value'].isin(removeList)]
    # Get Dummies, group by movie and aggregate the values as a sum in the rows
    dfMoviesCut = pd.get_dummies(
        dfMoviesCut, prefix="", prefix_sep="",).groupby(indexId).sum()
    # Replace everything bigger than 1 to a 1 -
    dfMoviesCut = dfMoviesCut.apply(lambda x: [1 if y >= 1 else 0 for y in x])

    dfMoviesCut.to_csv(csvSavePath)
    return dfMoviesCut

# %%
%%capture
dfcasts = createFeatureMatrix(r"./data/movies_casts.csv",
                              dfMoviesCastsTruncated, castColumns, removeListCasts)
dfcasts
# %%
%%capture
dfcrew = createFeatureMatrix(r"./data/movies_crew.csv",
                             dfMoviesCrewsTruncated, crewColumns, removeListCrews)
dfcrew
# %%
