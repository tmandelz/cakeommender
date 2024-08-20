# %% [markdown]

"""
# Datenbeschaffung
In diesem Teil werden die Daten für die Challenge beschafft.
Die Datensichtung und erste Bereinigungen wurden ebenfalls im Verlauf der Beschaffung durchgeführt.
"""

#%%
import os
if os.getcwd().endswith('tools'):
    os.chdir('..')
# %%
import platform
os = platform.system()

if os =="Linux":
    isNotpdfGenerator = False
else:
    isNotpdfGenerator = True

# %%
# Import der Libraries
import pandas as pd
import requests

# %% [markdown]

"""
## Funktionen
Für die Datenbeschaffung werden einige Funktionen benötigt.
Darunter beispielsweise eine Funktion welche tiefe JSON-Strukturen in flaches JSON umwandelt, damit dies nachher in ein pandas DataFrame eingefügt werden kann.
Auch der Download der Metadaten wurde in eine Funktion ausgelagert.
"""

# %%
# Credits to: https://stackoverflow.com/questions/52795561/flattening-nested-json-in-pandas-data-frame


def flattenJson(nestedJson, name, exclude=['']) -> dict:
    """
        Thomas Mandelz
        Function which flattens json object with nested keys into a single level.
        :param nestedJson: the nested json : json string
        :param name: prefix of the column names : str
        :param exclude: list keys to be excluded : list of strings
        :return: The flattened json object if successful, None otherwise.
        :rtype: dictionary
    """
    out = {}

    def flatten(x, name=name, exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude:
                    flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                if "job" in a:
                    if a.get("job") == "Director":
                        flatten(a, name + str(i) + '_')
                        i += 1
                else:
                    flatten(a, name + str(i) + '_')
                    i += 1
        else:
            out[name[:-1]] = x
    flatten(nestedJson)
    return out


def tmdbMetaDataDownload(csvSavePath: str, dfMovies: pd.DataFrame, topxRows: int = None, nestedColumnList=["genres", "production_companies", "production_countries", "spoken_languages"]) -> pd.DataFrame:
    """
        Thomas Mandelz
        Function loads metadata from the TMDB and combines them into a pandas dataframe
        :param csvSavePath: a path in r'path' format to save the dataframe as a csv file :str
        :param dfmovies: Dataframe with MovieLens Movies to gather Metadata: pd.DataFrame
        :param topxRows: an Integer to define if only the top x rows are downloaded :int
        :param nestedColumnList: Columns in the metadata which are nested : list of strings
        :return: Returns a full Dataframe of TMDB Metadata for all the movies
        :rtype: pd.DataFrame
    """
    # Create empty Dataframe
    dftTmdbMovies = pd.DataFrame()

    # Iteration Function
    def iteration(count: int, movieId: int, row: pd.Series, nestedColumnList: list) -> pd.DataFrame:
        # Extract tmdbId,title from current Row
        tmdbId = row["tmdbId"]
        title = row["title"]
        print(f"{count}: Movie: {tmdbId}, {title}")

        # Set up api URL and send Request
        apiUrl = f"https://api.themoviedb.org/3/movie/{tmdbId}?api_key=2e0eff00041b939a2244c9da913fb04f"
        response = requests.get(apiUrl)
        # Read response from json to pandas dataframe
        dfTemp = pd.json_normalize(response.json())
        # Add the MovieLensID to the df
        dfTemp["movieId"] = movieId
        # Flatten nested json inside pandas DF Columns
        for nestedColumn in nestedColumnList:
            try:
                # Flatten Json
                dfTemp = dfTemp.join(pd.DataFrame(
                    [flattenJson(x, nestedColumn) for x in dfTemp[nestedColumn]]))
                # Drop the column with nested json
                dfTemp = dfTemp.drop(columns=nestedColumn)
            except KeyError:
                # Some Rows are missing metadata, catch the Keyerror exception and ignore it
                print(f"Key {nestedColumn} not found! ignoring")
                pass

        return dfTemp

    count = 0
    # only "topxRows" are downloaded
    if topxRows == None:
        for i, row in dfMovies.iterrows():
            dfTemp = iteration(count, i, row, nestedColumnList)
            dftTmdbMovies = pd.concat([dftTmdbMovies, dfTemp])
            count += 1
    # All rows are downloaded
    else:
        for i, row in dfMovies.head(topxRows).iterrows():
            dfTemp = iteration(count, i, row, nestedColumnList)
            dftTmdbMovies = pd.concat([dftTmdbMovies, dfTemp])
            count += 1
    # Join MovieLens and TMDB Metadata Dataframe with each other
    dftTmdbMovies = pd.concat([dftTmdbMovies.set_index(
        'movieId'), dfMovies], axis=1, join='inner')
    # Save the joined Dataframe to a csv
    dftTmdbMovies.to_csv(csvSavePath)

    return dftTmdbMovies


def crewCastMetadataDownload(csvSavePath: str, dfmovies: pd.DataFrame, topxRows: int = None, nestedColumnList=["cast", "crew"]) -> pd.DataFrame:
    """
        Thomas Mandelz
        Function loads metadata from actors and crew from the TMDB and combines them into a pandas dataframe
        :param csvSavePath: a path in r'path' format to save the dataframe as a csv file :str
        :param dfmovies: Dataframe with MovieLens Movies to gather Metadata: pd.DataFrame
        :param topxRows: an Integer to define if only the top x rows are downloaded :int
        :param nestedColumnList: Columns in the metadata which are nested : list of strings
        :return: Returns a full Dataframe of TMDB Metadata for all the movies
        :rtype: pd.DataFrame
    """
    dfTmdbCast = pd.DataFrame()
    # Iteration Function

    def iteration(count: int, movieId: int, row: pd.Series, nestedColumnList: list) -> pd.DataFrame:
        # Extract tmdbId,title from current Row
        tmdbId = row["tmdbId"]
        title = row["title"]
        print(f"{count}: Cast & Crew for Movie: {tmdbId}, {title}")

        # Set up api URL and send Request
        apiUrl = f"https://api.themoviedb.org/3//movie/{tmdbId}/credits?api_key=2e0eff00041b939a2244c9da913fb04f"
        response = requests.get(apiUrl)
        # Read response from json to pandas dataframe
        dfTemp = pd.json_normalize(response.json())

        # Add the MovieLensID to the df
        dfTemp["movieId"] = movieId
        # Flatten nested json inside pandas DF Columns
        for nestedColumn in nestedColumnList:
            try:
                # Flatten Json
                dfTemp = dfTemp.join(pd.DataFrame(
                    [flattenJson(x, nestedColumn) for x in dfTemp[nestedColumn]]))
                # Drop the column with nested json
                dfTemp = dfTemp.drop(columns=nestedColumn)
            except KeyError:
                # Some Rows are missing metadata, catch the Keyerror exception and ignore it
                print(f"Key {nestedColumn} not found! ignoring")
                pass

        return dfTemp

    count = 0
    # only "topxRows" are downloaded
    if topxRows == None:
        for i, row in dfmovies.iterrows():
            dfTemp = iteration(count, i, row, nestedColumnList)
            dfTmdbCast = pd.concat([dfTmdbCast, dfTemp])
            count += 1
    # All rows are downloaded
    else:
        for i, row in dfmovies.head(topxRows).iterrows():
            dfTemp = iteration(count, i, row, nestedColumnList)
            dfTmdbCast = pd.concat([dfTmdbCast, dfTemp])
            count += 1
    # Join MovieLens and TMDB Metadata Dataframe with each other
    dfTmdbCast = pd.concat([dfTmdbCast.set_index(
        'movieId'), dfmovies], axis=1, join='inner')
    # Save the joined Dataframe to a csv
    dfTmdbCast.to_csv(csvSavePath)

    return dfTmdbCast


# %% [markdown]

"""
## MovieLens
Als erstes werden die MovieLens CSV-Dateien für die Filme und deren Links (tdmbID und imdbID) eingelesen.
"""

# %%
dfMovies = pd.read_csv(r"./movielens_data/movies.csv")
dfLinks = pd.read_csv(r"./movielens_data/links.csv")

# %% [markdown]
"""
Zunächst wird das **movies**-DataFrame inspiziert. Die ersten Zeilen werden ausgegeben und visuell überprüft.
"""
# %%
dfMovies.head()

# %% [markdown]
"""
Weiter wird das **links**-DataFrame analysiert. Die ersten Zeilen davon sehen folgendermassen aus:
"""
# %%
dfLinks.head()

# %% [markdown]
"""
### MovieLens NA-Überprüfung
Bevor wir weiter mit den MovieLens-Daten arbeiten, prüfen wir zuerst auf etwaige NA-Werte.
"""

# %%
print('Anzahl NAs in Attribut movieId von dfMovies:', dfMovies['movieId'].isna().sum())
print('Anzahl NAs in Attribut title von dfMovies:  ', dfMovies['title'].isna().sum())
print('Anzahl NAs in Attribut genres von dfMovies: ', dfMovies['genres'].isna().sum())
print('Anzahl NAs in Attribut movieId von dfLinks: ', dfLinks['movieId'].isna().sum())
print('Anzahl NAs in Attribut imdbId von dfLinks:  ', dfLinks['imdbId'].isna().sum())
print('Anzahl NAs in Attribut tmdbId von dfLinks:  ', dfLinks['tmdbId'].isna().sum())

# %%
print('NAs in Spalte tmdbId von dfLinks:')
dfLinks[dfLinks['tmdbId'].isna()].head()

# %% [markdown]
"""
Nur in der `tmdbId`-Spalte bei den Links kommen NA-Werte vor.
Da wir ohne den Fremdschlüssel `tmdbId` keine Metadaten von TMDB herunterladen können und diesen nicht bei 181 Filmen manuell imputieren wollen, entfernen wir diese Filme.
Beim Einlesen dieser Spalte `tmdbId` hat pandas de Datentyp `float` für die seID gewählt, da diese NA-Werte enthielt.
Die ist durch das Entfernen nicht mehr nötig und wir können diesen auf `int` ändern.
"""

# %%
dfLinks = dfLinks[~dfLinks['tmdbId'].isna()]
dfLinks['tmdbId'] = dfLinks['tmdbId'].astype(int)
dfLinks.head()

# %% [markdown]
"""Nun scheint diese Spalte auch sauber zu sein und wir können fortfahren."""

# %% [markdown]
"""
### Verlinkung MovieLens mit TMDB
Glücklicherweise ist im MovieLens-Datensatz bereits der Fremdschlüssel zur **The Movie Database** beinhaltet.
Da die Links jedoch in einem separaten DataFrame abgelegt sind, werden sie zuerst mittels einem Inner-Join in ein vollumfängliches DataFrame kombiniert.
Danach prüfen wir auch hier den Join visuell indem wir die einige Zeilen ausgeben.
"""
# %%
dfMovies = pd.concat([
    dfMovies.set_index('movieId'), dfLinks.set_index('movieId')
], axis=1, join='inner')

# %%
dfMovies.head()

# %% [markdown]
"""
## Regex Filter für Datum
Da wir nicht den kompletten Filmdatensatz für unser Recommendersystem verwenden möchten, grenzen wir diesen ein auf die Filme, die im Jahr 2010 oder später veröffentlicht wurden.
Das Veröffentlichungsjahr ist bei MovieLens im Titel des Films eingebettet.
<br>
Beispiel: Toy Story **(1995)**
<br>
Mittels der Regex-Abfrage `\((201[1-9]\d|20[1-2]\d)\)` können wir die Filme selektieren, die dem Ausdruck entsprechen. So können wir alle Filme herausfiltern dessen Jahreszahl nicht >= 2010 entspricht.
"""

# %%
import warnings
warnings.filterwarnings("ignore", 'This pattern is interpreted as a regular expression, and has match groups.')
# Remove all movies older than 2010 with a regex
dfMoviesTruncated = dfMovies[dfMovies.title.str.contains('\((201[1-9]\d|20[1-2]\d)\)')]

print('Anzahl entfernter Filme:', len(dfMovies) - len(dfMoviesTruncated))
print('Anzahl verbleibender Filme:', len(dfMoviesTruncated))

# %% [markdown]
"""
## Download der Filmmetadaten von TMDB
Die Metadaten werden anhand der TMDBID von der TMDB-API heruntergeladen und mit dem MovieLens-Datensatz verknüpft.
Das Resultat ist ein DataFrame, in welchem die Daten in einer flachen Struktur abgelegt sind und das alle Metadaten sowie die MovieLens-Daten beinhaltet.
Dieses DataFrame wird als CSV-Datei zwischengespeichert.
<br>
Zuerst führen wir einen kurzen Testlauf aus mit nur 20 Zeilen, um die korrekte Funktionsweise der Funktion zu prüfen.
"""
# %%
dfTmdbMovies = tmdbMetaDataDownload(
    r"./data/movies_temp.csv", dfMoviesTruncated, 20)

# %% [markdown]
"""
Eine visuelle Inspektion der vorhandenen Spalten ist auch hier durchzuführen.
"""

# %%
dfTmdbMovies.columns.values

# %% [markdown]
"""
Da das Resultat im Testlauf gut aussieht können wir nun den kompletten Download durchführen.
"""

# %%
# To minimize notebook runtime, this was run once and then commented out
if isNotpdfGenerator:
    dfTmdbMovies = tmdbMetaDataDownload(r"./data/movies.csv", dfMoviesTruncated)

# %% [markdown]
"""
## Download des Cast und der Crew von TMDB
Die Schauspieler und Regisseure der Filme werden nun anhand der TMDBID von der TMDB-API heruntergeladen und auch mit dem MovieLens-Datensatz verknüpft.
Das resultierende DataFrame enthält auch diese Daten in einer flachen Struktur.
<br>
Wir testen die Funktion und laden dazu Daten zu 20 Filmen herunter und legen diese in einer CSV-Datei ab.
"""

# %%
dfTmdbCast = crewCastMetadataDownload(
    r"./data/movies_casts_crew_temp.csv", dfMoviesTruncated, 20)

# %% [markdown]
"""
Die heruntergeladenen Daten enthalten diese Spalten:
"""
# %%
dfTmdbCast.columns.values

# %% [markdown]
"""
Nun da das Resultat im Testlauf gut aussieht können wir den kompletten Download des Cast und der Regisseure durchführen.
"""

# %%
# To minimize notebook runtime, this was run once and then commented out
if isNotpdfGenerator:
    dfTmdbCast = crewCastMetadataDownload(r"./data/movies_casts_crew.csv", dfMoviesTruncated)

# %%
