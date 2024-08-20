#%% [markdown]
## User für Evaluation
"""
Dieses Notebook wurde verwendet um Nutzer für qualitative Auswertungen herauszusuchen.
Dabei werden immer zwei Nutzer als Vergleichskunden herausgesucht.
Es wird versucht mit einer optimalen Auswahl der Nutzer, das Model qualitativ möglichst gut zu prüfen.
Darum werden die Nutzer so ausgewählt, dass sie möglichst viele Kunden repräsentieren.
Die Nutzer können in folgende Kategorien unterteilt werden:
- Wie gut ein Rating von einem User ist.
- wie spezifisch das Interesse des Nutzers ist.
"""

#%% [markdown]
"""
Aus diesen Kategorien werden möglichst unterschiedliche Kombinationen gewählt.
"""

#%%
import os
if os.getcwd().endswith('eda'):
    os.chdir('..')

#%% [markdown]
## Libraries laden und Daten einlesen
# %% 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
allRatings = pd.read_csv(r"./data/userItem_stand_long.csv")
movieGenre = pd.read_csv(r"./data/movies_genres.csv")
moviesMeta = pd.read_csv(r"./data/movies_meta.csv")

# %%
allRatings["userId"] = allRatings["userId"].astype(int)

# count rated movies
countRatings = allRatings.groupby(['userId'])["movieId"].count().reset_index().rename(columns={"movieId": "count"})
ratings= allRatings.merge(countRatings,how = "left",on = "userId")
# %%
# join ratings with metadata
metaRatings = ratings.merge(moviesMeta,how = "inner",on = "movieId")
genreRatings = ratings.merge(movieGenre,how = "inner",on = "movieId")

#%% [markdown]
## Kunden mit vielen Bewertungen
"""
Die Verteilung der Anzahl Ratings pro User ist stark rechtsschief. Um User herauszufiltern, die viele Ratings abgegeben haben, nehmen wir das 50% Quantil der Verteilung als Grenze an. Das heisst konkret, dass 50% der User viele Ratings abgegeben haben. Dieser Wert liegt bei etwas über 100 Ratings pro User.
"""
# %%
sns.kdeplot(ratings["count"])
plt.title("Anzahl user: " + str(len(pd.unique(ratings["userId"]))))
plt.suptitle("Verteilung der Anzahl Ratings pro User")
plt.vlines(ratings["count"].quantile(0.5),0,0.008,colors = "red")
plt.xlabel("Anzahl der Ratings pro User")
plt.ylabel("Verteilung")
plt.show()
# %% [markdown]
## Zwei Kunden mit spezifischen, unterschiedlichen Interessen
"""
Es werden Kunden gesucht mit einem sehr spezifischen, jedoch unterschiedlichem, Interesse und vielen Bewertungen.
"""

#%%
genreUserManyRatings = genreRatings[genreRatings["count"] > ratings["count"].quantile(0.5)]
metaUserManyRatings = metaRatings[metaRatings["count"] > ratings["count"].quantile(0.5)]
# %% [markdown]
"""
Folgende User werden im nächsten Abschnitt als Vergleichskunden ausgewählt:
- User der romantische Filme mag
- User der Action-Filme mag
"""
# %%
actionUser = pd.DataFrame(genreUserManyRatings[genreUserManyRatings["Action"] == 1].groupby(["userId"])["rating"].mean())
romanceUser = pd.DataFrame(genreUserManyRatings[genreUserManyRatings["Romance"] == 1].groupby(["userId"])["rating"].mean())
actionRomance = actionUser.merge(romanceUser,how = "inner",on = "userId")

user1Id = actionRomance[actionRomance["rating_x"]==min(actionRomance["rating_x"])].index.values[0]
user1Description = "Actionfilm-Liebhaber"
user2Id = actionRomance[actionRomance["rating_y"]==max(actionRomance["rating_y"])].index.values[0]
user2Description = "Romantikfilm-Liebhaber"

userEvaluation = pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})

# %% [markdown]
"""
Folgende User werden im nächsten Abschnitt als Vergleichskunden ausgewählt:
- User, welcher Filme schaut, die kürzer als 90 Minuten gehen.
- User, welcher Filme schaut, die länger als 120 Minuten gehen.
"""
# %%
metaUserManyRatings.loc[:,"short_movies"] = metaUserManyRatings.loc[:,"runtime"] < 90
metaUserManyRatings.loc[:,"long_movies"] = metaUserManyRatings.loc[:,"runtime"] > 120
shortRuntimeUser = pd.DataFrame(metaUserManyRatings[metaUserManyRatings["short_movies"]].groupby(["userId"])["rating"].mean())
longRuntimeUser = pd.DataFrame(metaUserManyRatings[metaUserManyRatings["long_movies"]].groupby(["userId"])["rating"].mean())

user1Id = shortRuntimeUser[shortRuntimeUser["rating"]== max(shortRuntimeUser["rating"])].index.values[0]
user1Description = "User, der lange Filme mag"
user2Id = longRuntimeUser[longRuntimeUser["rating"]== max(longRuntimeUser["rating"])].index.values[0]
user2Description = "User, der kurze Filme mag"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))

# %% [markdown]
"""
Folgende User werden im nächsten Abschnitt als Vergleichskunden ausgewählt:
- User, der oft bewertete Filme mag: Blockbuster - Liebhaber
- User, der wenig bewertete Filme mag: Raritäten - Jäger
"""
#%%
ratings = ratings.merge(allRatings.groupby("movieId").size().to_frame('popularity'),how = "left", on = "movieId")
# %%
ratings.loc[:,"Unpopular"] = ratings["popularity"] < 500
ratings.loc[:,"Popular"] = ratings["popularity"] > 1000
popularMovieUser = pd.DataFrame(ratings[ratings["Popular"]].groupby(["userId"])["rating"].mean())
unpopularMovieUser = pd.DataFrame(ratings[ratings["Unpopular"]].groupby(["userId"])["rating"].mean())

user1Id = popularMovieUser[popularMovieUser["rating"]== max(popularMovieUser["rating"])].index.values[0]
user1Description = "User, der populäre Filme mag"
user2Id = unpopularMovieUser[unpopularMovieUser["rating"]==max(unpopularMovieUser["rating"])].index.values[0]
user2Description = "User, der unpopuläre Filme mag"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))

# %% [markdown]
## Zwei Kunden mit gleichen Interessen
"""
Es werden Kunden gesucht, die ein ähnliches spezifisches Interesse haben.
"""
# %% [markdown]
"""
- Es werden zwei Musikfilm-Liebhaber gesucht.
"""

# %%
musicUser = pd.DataFrame(genreUserManyRatings[genreUserManyRatings["Music"]==1].groupby(["userId"])["rating"].mean()).nlargest(2,"rating")

user1Id = musicUser.index[0]
user1Description = "Musikfilm - Liebhaber"
user2Id = musicUser.index[1]
user2Description = "Musikfilm - Liebhaber"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))

# %% [markdown]
"""
- Es werden zwei Horrorfilm-Liebhaber gesucht.
"""
# %%
horrorUser = pd.DataFrame(genreUserManyRatings[genreUserManyRatings["Horror"]==1].groupby(["userId"])["rating"].mean()).nlargest(2,"rating")

user1Id = horrorUser.index[0]
user1Description = "Horrorfilm - Liebhaber"
user2Id = horrorUser.index[1]
user2Description = "Horrorfilm - Liebhaber"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))

# %% [markdown]
## Spezifisches Interesse vs offener User
"""
Folgende User werden im Nächsten als Vergleichskunden ausgewählt:
- User mit einem spezifischen Interesse
- User, welcher sehr unterschiedliche Filme schaut
"""
# %%
summedGenres = genreUserManyRatings.drop(["Unnamed: 0","movieId","rating","count"],axis = 1).groupby(["userId"]).sum()
# %%
# User mit der kleinsten Varianz wird berechnet
varianceGenres = summedGenres.std(axis = 1)

user1Id = varianceGenres.nsmallest(1).index.values[0]
user1Description = "offenes Interesse"
user2Id = varianceGenres.nlargest(1).index.values[0]
user2Description = "spezifisches Interesse"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))
# %% [markdown]
## Zwei User mit offenem Interessen
"""
Es werden zwei User herausgesucht, welche ihre Filme divers aussuchen und somit auch ein Interesse haben, unterschiedliche Filme angeboten zu bekommen. Die Diversität der Filme wird anhand von unterschiedlichen Genres gemessen.
"""
# %%
user1Id = varianceGenres.nsmallest(2).index.values[1]
user1Description = "offenes Interesse"
user2Id = varianceGenres.nsmallest(3).index.values[2]
user2Description = "offenes Interesse"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))
# %% [markdown]
## Zwei User die wenig Ratings abgegeben haben
"""
Es werden gezielt zwei User ausgewählt, die sehr wenige Bewertungen abgegeben haben.
Dadurch kann geprüft werden, ob das Modell bei diesen Usern funktioniert, oder ob die minimale Anzahl von bewerteten Filme höher gesetzt werden sollte.
"""
# %%
countAllRatings = allRatings.groupby(['userId']).count()

user1Id = countAllRatings.nsmallest(1,"rating").index.values[0]
user1Description = "wenige Ratings"
user2Id = countAllRatings.nsmallest(2,"rating").index.values[1]
user2Description = "wenige Ratings"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))
# %% [markdown]
## Viel bewertet vs wenig bewertet
"""
Folgende User werden im nächsten Abschnitt als Vergleichskunden ausgewählt:
- Es wird ein User ausgewählt, der sehr wenige Bewertungen abgegeben haben.
- Es wird ein User ausgewählt, der viele Bewertungen abgegeben haben.
"""
# %%
user1Id = countAllRatings.nsmallest(3,"rating").index.values[2]
user1Description = "wenige Ratings"
user2Id = countAllRatings.nlargest(1,"rating").index.values[0]
user2Description = "viele Ratings"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))
# %% [markdown]
## Zwei Standard-User
"""
So gut wie möglich sollte ein Standard-User herausgesucht werden. Diese Person schaut immer wiedermal einen Film und ist nicht alzu wählerisch. Trotzdem gibt es gewisse Filme, die diese Person nicht mag.
"""
# %%
# User Id's for "normal" active raters
quantilesRatingCount = countAllRatings["rating"].quantile([0.47,0.53]).values
normalRatersCount = countAllRatings.loc[(quantilesRatingCount[0] < countAllRatings["rating"]) & (quantilesRatingCount[1] > countAllRatings["rating"]),"rating"]
# %%
# User Id's with a "normal" variance
fullDataVarianceGenres = genreRatings.drop(["Unnamed: 0","movieId","rating","count"],axis = 1).groupby(["userId"]).sum().std(axis = 1).reset_index()
quantileRatersVariance = fullDataVarianceGenres[0].quantile([0.47,0.53]).values
normalRatersVariance = fullDataVarianceGenres[(quantileRatersVariance[0] < fullDataVarianceGenres[0]) & (quantileRatersVariance[1] > fullDataVarianceGenres[0])]
# %%
normalUser= normalRatersVariance.merge(normalRatersCount,how = "inner",on = "userId")["userId"].sample(2, random_state=53).values

user1Id = normalUser[0]
user1Description = "normaler User"
user2Id = normalUser[1]
user2Description = "normaler User"

userEvaluation = pd.concat((userEvaluation,pd.DataFrame({"User1Id":[user1Id],"User1Description":[user1Description],"User2Id":[user2Id] ,"User2Description":[user2Description]})))

# %%
userEvaluation.to_csv("./data/userQualitativeEvaluation.csv",index=False)
# %%
