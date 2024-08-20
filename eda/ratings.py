# %% [markdown]
"""
# Datenanalyse für Ratings
Dieser letzte Teil wird verwendet, um die Ratings aller User zu reduzieren und in die Form einer User-Item-Matrix zu bringen. Die hierzu verwendeten Ratingskalen entsprechen der Projektvereinbarung: binäre, tertiäre und standardisierte Ratingskala. Zur Erstellung der Ratingskalen werden die Ratings pro standardisiert und ein spezifischer Grenzwert bestimmt anstelle von fixen Bewertungsgrenzen.
"""

#%%
import os
if os.getcwd().endswith('eda'):
    os.chdir('..')

#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
if os.getcwd().endswith('eda'):
    os.chdir('..')

# %% [markdown]
"""
## Einlesen der Daten
Die Rating-Daten von der MovieLens-Datenbank und die für das Recommendersystem verwendeten Filme werden als erstes eingelesen.
"""

#%%
%%capture
ratings = pd.read_csv("movielens_data/ratings.csv")

#%%
%%capture
movies = pd.read_csv("data/movies_meta.csv")

# %% [markdown]
"""
## Reduktion des Datensatzes
### Recommender-Filme
Wir möchten nur Ratings behalten, für die es auch einen Film in unseren aufbereiteten Filmdaten gibt.
"""

#%%
movies = movies[['movieId', 'release_date']]

#%%
rates = movies.merge(ratings, on='movieId', how='left')

# %% [markdown]
"""
### Sinnvolle Ratings
Nur die nach dem Filmveröffentlichungsdatum erstellten Kommentare machen Sinn. Alle anderen werden aus dem Datensatz gelöscht.
"""

#%%
rates['release_date']=pd.to_datetime(rates['release_date'])
rates['timestamp']=pd.to_datetime(rates['timestamp'], unit='s').dt.date

#%%
rates['flag'] = np.where(rates['release_date'] <= rates['timestamp'], 1, 0)

#%%
print('Anzahl entfernter Ratings:', (rates.flag == 0).sum())
rates = rates[rates.flag != 0]
rates.drop(['flag', 'timestamp', 'release_date'],axis=1, inplace=True)

# %% [markdown]
"""
### Reduktion User
Wir entfernen User, die weniger als 3 Ratings abgegeben haben aus dem Datensatz. Da wir jeden neuen Nutzer bitten, 3 Lieblingsfilme anzugeben, sind Nutzerprofile sinnvollerweise aus mindestens 3 Ratings zu generieren.
Es verbleiben 58'945 User mit 2.3 Mio. Ratings.
"""
#%%
rates['userId'] = rates['userId'].astype(int)
rates_red0 = rates.groupby('userId').filter(lambda x: len(x)>=3)

# %% [markdown]
"""
### Standardisieren
Wir standardisieren unsere Ratings pro User bevor wir diese nochmals reduzieren:

$$r_i = \frac{r_i - \mu_u}{\sigma_u}$$
"""

#%%
rates_stand = rates_red0.groupby('userId')
rates_red0['rating_st'] = rates_stand['rating'].transform(lambda x: (x - x.mean()) / x.std())

# %% [markdown]
"""
User, die nur immer die gleichen Bewertungen abgeben, werden wir ausschliessen. Dies insbesondere, weil unsere binären und tertiären Ratingskalen auf der standardisierten Skala basieren und nur überdurchschnittliche Filme als gut bewerten werden. Das heisst konkret, dass alle Filme von einem User, der nur 5er Ratings abgegeben hat, als schlecht qualifiziert werden. Dies wäre natürlich eine Verfälschung, denn er fand die Filme ja alle sehr gut. Uns ist bewusst, dass wir dieses Problem mit einer anderen Art von binären und tertiären Ratingskala lösen könnten. Aufgrund unserer Ressourcen und da es nicht viele Ratings betrifft, werden wir uns im Rahmen dieser Challenge jedoch nicht auf diese Problematik konzentrieren.
"""

#%%
print('Anzahl entfernter Ratings:', rates_red0['rating_st'].isna().sum())
rates_red0.dropna(inplace=True)

# %% [markdown]
"""
### Reduktion auf 10'000 User
Aus den verbliebenen 57'456 Usern, reduzieren wir den Datensatz auf 10'000 zufällig ausgesuchte User. Damit werden später Nutzerprofile erstellt. Aus unserer Sicht benötigt es nicht mehr als 10'000 Nutzerprofile, da wir ein content-based Recommendersystem aufbauen und diese Nutzerprofile lediglich für die Evaluierung benötigen. Wir überprüfen die Verteilung der standardisierten Ratings zwischen dem auf 10'000 User reduzierten und den mit knapp 59'000 User vollständigen Datensatz.
"""

#%%
np.random.seed(20)
g = rates_red0.groupby('userId')
a= np.arange(g.ngroups)
np.random.shuffle(a)
rates_red = rates_red0[g.ngroup().isin(a[:10000])]
rates_stand = rates_red.drop(['rating'], axis=1)
q6 = rates_stand.rating_st.quantile(0.6)
q6_per_user = rates_stand.groupby('userId').rating_st.quantile(0.6).rename('q6')
rates_stand = pd.merge(rates_stand, q6_per_user, left_on='userId', right_index=True)

# %% [markdown]
"""
Die Verteilung des gesamten und des reduzierten Datensatzes ist sehr ähnlich.
"""

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle("Verteilung der Ratings")

ax1.hist(rates_red0.rating_st, range=[-5,5], bins=50)
ax1.set_title("Gesamter Datensatz, n=2'317'982")
ax1.set_xlabel("Standardisiertes Rating pro User")
ax1.set_ylabel("Anzahl Ratings")

ax2.hist(rates_stand.rating_st, range=[-5,5], bins=50)
ax2.set_title("Reduzierter Datensatz, n=394'991")
ax2.set_ylabel("Anzahl Ratings")
ax2.set_xlabel("Standardisiertes Rating pro User")
plt.show()

# %% [markdown]
"""
## Ratingskalen
Gemäss unserer Projektvereinbarung möchten wir drei verschiedene Rating-Skalen für unser Recommendersystem nutzen.

- Standardisierte Ratingskala: kontinuierliches, auf den User standardisiertes Rating

- Binäre Ratingskala: Alle Ratings mit einem grösseren oder gleichen Wert wie des standardiserten 60%-Quantil pro User werden als gute Filme kategorisiert (1), alle anderen Filme als schlecht (0). Nicht geratete Filme werden ebenfalls mit 0 bewertet.

- Tertiäre Ratingskala: Analog binäre Ratingskala, schlechte Filme werden jedoch als -1 gewertet, um sie von nicht bewerteten Filmen unterscheiden zu können.
"""

# %% [markdown]
"""
### Binäre Ratingskala
"""

#%%
#Guter Film, sofern über 60% percentil pro User, sonst 0
rates_stand['bin'] = np.where(rates_stand.rating_st >= rates_stand.q6, 1, 0)

# %% [markdown]
"""
In der folgenden Grafik ist erkennbar, wie die binären Ratings verteilt sind. Der Überlappungsbereich ist relativ klein.
"""

# %%
plot = rates_stand.pivot(columns='bin', values='rating_st')
fig, (ax10, ax20) = plt.subplots(1, 2, figsize=(12,5))
fig.suptitle("Verteilung der Ratings, Reduzierter Datensatz, n=394'991")

ax10.hist(rates_stand.rating_st, range=[-5,5], bins=100)
ax10.set_title("Standardisiertes Rating")
ax10.set_ylabel("Anzahl Ratings")
ax10.set_xlabel("Standardisiertes Rating pro User")

ax20.hist(plot, range=[-5,5], bins=100)
ax20.set_title("Binäres Rating")
ax20.set_ylabel("Anzahl Ratings")
ax20.set_xlabel("Standardisiertes Rating pro User")
ax20.legend(['schlecht', 'gut'])
plt.show()

# %% [markdown]
"""
### Tertiäre Ratingskala
Die Verteilung der tertiären Ratingskala ist analog zur binären, nur dass die als schlecht klassifizierten Filme einen Wert von -1 erhalten.
"""

#%%
# Guter Film, sofern über 60% percentil, sonst -1
rates_stand['tert'] = np.where(rates_stand.rating_st >= rates_stand.q6, 1, -1)

# %% [markdown]
"""
## Export
### User-Item Matrix für Nutzerprofile
Im folgenden werden die Ratings der drei Skalen je in die Form einer User-Item-Matrix gebracht. Da die Matrix sparse ist, werden die NAs nicht mit 0 aufgefüllt, um die zu exportierenden CSV-Dateien möglichst klein zu halten.
"""

#%%
rates_stand_pivot = pd.pivot(rates_stand, index='movieId', columns = 'userId', values='rating_st')
rates_bin = pd.pivot(rates_stand, index='movieId', columns = 'userId', values='bin')
rates_tert = pd.pivot(rates_stand, index='movieId', columns = 'userId', values='tert')

#%%
rates_stand_pivot.to_csv(r"./data/userItem_stand.csv")
rates_bin.to_csv(r"./data/userItem_bin.csv")
rates_tert.to_csv(r"./data/userItem_tert.csv")

# %% [markdown]
"""
### Export für Evaluation
Wir exportieren die User-Item-Matrizen ebenso im langen Format für alle Ratingskalen. Diese benötigen wir für die Evaluierung.
"""

#%%
%%capture
rates_binlist = rates_stand[['movieId', 'userId', 'bin']]
rates_binlist.rename(columns={'bin': 'rating'}, inplace=True)
rates_binlist.to_csv(r"./data/userItem_bin_long.csv")
# %%
%%capture
rates_tertlist = rates_stand[['movieId', 'userId', 'tert']]
rates_tertlist.rename(columns={'tert': 'rating'}, inplace=True)
rates_tertlist.to_csv(r"./data/userItem_tert_long.csv")

#%%
%%capture
rates_stanlist = rates_stand[['movieId', 'userId', 'rating_st']]
rates_stanlist.rename(columns={'rating_st': 'rating'}, inplace=True)
rates_stanlist.to_csv(r"./data/userItem_stand_long.csv")


