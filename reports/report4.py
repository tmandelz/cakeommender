#%% [markdown]
# \begin{titlepage}
#    \begin{center}
#       \vspace*{1cm}
#       \Huge
#       \textbf{Report 4}

#       \vspace{0.5cm}
#       Tinder for Movies - Challenge HS 2022

#       \vspace{1.5cm}

#       \Large
#       Daniela Herzig, Thomas Mandelz, Joseph Weibel, Jan Zwicky

#       \normalsize
#       Januar 2023

#       \vfill

#       \tiny
#    \end{center}
# \end{titlepage}
# \tableofcontents
# \newpage
# %%
import os
if os.getcwd().endswith('reports'):
	os.chdir('..')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libraries.matrix import MatrixGenerator
from libraries.pipeconfig import (
	CakeConfig,
	FeatureNormalizationEnum,
	SimilarityEnum,
	RatingScaleEnum,
)
from libraries.weights import WeightGenerator
from cakeommender import Cakeommender
from IPython.display import display, Markdown

pd.set_option("display.notebook_repr_html", False)


# %% [markdown]
"""
In diesem Teil werden wir unser bestes Recommendersystem in einen Prototypen einbauen und die Filmvorschläge für einige Nutzende überprüfen.

# Bestes Modell

Im letzten Teil unserer Arbeit haben wir unterschiedliche Recommendersysteme evaluiert und konnten eine Kombination von Film-Metadaten und Sentence-Embeddings als unser bestes bestimmen. Die Sentence-Embeddings wurden auf Basis der kurzen Filmbeschreibung mit SBERT generiert. Die  Features wurden standardisiert und folgendermassen gewichtet:

| **Feature**              | **Gewichtung** |
|--------------------------|----------------|
| Filmbudget               | 1.0            |
| Umsatz                   | 1.0            |
| Filmlänge                | 1.0            |
| Genres                   | 1.0            |
| Schauspieler*innen       | 0.4            |
| Filmbeschreibung (SBERT) | 0.6            |

"""

# %% [markdown]
"""
Regisseur*innen wurden nicht in die Featurematrix aufgenommen. Zur Berechnung der Ähnlichkeit verwenden wir das Cosine-Ähnlichkeitsmass und wandeln die Ratings in die tertiäre Skala (-1, 0, 1) um.

# Funktionsweise Prototyp

Dieses Recommendersystem haben wir nun in einen Prototypen eingebaut. Dieser soll es zwei Nutzer*innen ermöglichen passende Filmempfehlungen für einen gemeinsamen Filmabend oder ähnlichem zu finden, ohne dass mühselig von Hand Filme durchgearbeitet werden müssen, bis man einen Film gefunden hat, den beide mögen könnten. Die Nutzenden können jeweils voneinander unabhängig ihnen bekannte Filme auswählen, die ihnen sehr gut gefallen haben und auch solche, die sie nicht weiterempfehlen würden. Die Anwendung stellt den beiden anschliessend eine Top-5-Liste mit Filmempfehlungen zusammen, die mit unserem besten Recommendersystem berechnet werden. Es werden dabei die Wünsche der beiden zu gleichem Mass berücksichtigt, so dass unter diesen Empfehlungen möglichst nur Filme vorkommen, die für beide infrage kommen.

![Screenshot des Prototyps](reports/res/app_overview.png)

Im Prototyp sind dazu pro Nutzenden zwei Eingabefelder vorhanden, in die einerseits beliebte und im anderen unbeliebte Filme eingetragen werden können. Dabei kann nach Filmtiteln gesucht werden und die Anwendung liefert während der Eingabe ihr bekannte Filmtitel, die dann mittels Pfeiltasten oder Maus ausgewählt werden können. Sobald mindestens drei Filme ausgewählt wurden, kann das Recommendersystem gestartet werden und die Empfehlungen werden unterhalb mit Titel und Filmbeschreibung angezeigt. Die Kuchenbilder spielen dabei auf den Namen unseres Projekts an und haben keinen Zusammenhang mit den Filmen.

## Technischer Aufbau

Die Anwendung wurde als Web-App umgesetzt, so dass sie direkt im Browser verwendet werden kann. Als Framework wurde *Dash* eingesetzt, dass die HTML-Struktur generiert und die Kommunikation zwischen Browser und Server abstrahiert. Für das Aussehen wurde mit CSS-Stylesheets und CSS-Anweisungen direkt im Dash-Code gearbeitet.

Die Filmempfehlungen werden in einem Dash-Callback berechnet. Beim Absenden des Formulars wird dieser mit den ausgewählten Filmen aufgerufen, wo das Nutzerprofil anhand der ausgewählten Filme zusammengestellt wird. Die daraus resultierenden Empfehlungen werden im Callback zurückgegeben und unterhalb der Kuchengrafiken eingesetzt.

# Kombinationsmöglichkeiten

Um die ausgewählten Filme der beiden Nutzenden kombinieren zu können und daraus Empfehlungen ableiten zu können, bieten sich zwei Varianten an. So können die ausgewählten Filme pro User in ein Nutzerprofil umgewandelt und für jedes dieser Profile die Filmempfehlungen separat berechnet werden. Die gemeinsamen Empfehlungen ergeben sich dann daraus, indem Überschneidungen zwischen den zwei Top-N-Listen gefunden werden. Diese Überschneidungen müssen für eine gemeinsame Top-N-Liste wieder sortiert werden, was anhand des gemittelten Ähnlichkeitswerts aus den beiden ursprünglichen Top-N-Listen gemacht werden kann. Um eine fixe Anzahl Empfehlungen mit dieser Variante zu erhalten, kann in der Praxis wohl nicht mit fixen Top-N-Listen als Grundlage für die Kombination gearbeitet werden, da das N nicht festgelegt werden kann. Bei sehr unterschiedlichen Nutzerprofilen wird es wenige Überschneidungen in den oberen Rängen von diesen Listen geben. So dass um eine fixe Anzahl an Überschneidungen zu erhalten, als Ausgangslage alle noch nicht gesehenen Filme pro User mit ihren Ähnlichkeitswerten zum Nutzerprofil verwendet werden müssen.

Die zweite Möglichkeit besteht darin, die beiden Nutzerprofile zu summieren, bevor sie für die Ähnlichkeitsberechnung verwendet werden. Die Profile ($u_1$ und $u_2$) werden zuerst normiert auf Betrag 1 und anschliessend elementweise aufaddiert. Durch die Normierung erhalten beide Nutzerprofile den gleichen Einfluss auf das kombinierte Profil. Ein Profil, das aus mehr Filmen als das andere Profil aufgebaut ist, hätte ansonsten einen grösseren Einfluss gehabt als das zweite.

\begin{equation*}
u_{combined} = \frac{u_1}{||u_1||_2} + \frac{u_2}{||u_2||_2}
\end{equation*}

Dieser Ansatz könnte auch für eine beliebige Anzahl an Nutzenden ($n$) erweitert werden:

\begin{equation*}
u_{combined} = \sum_{i=1}^{n} \frac{u_i}{||u_i||_2}
\end{equation*}

Mit dieser Variante können keine Empfehlungen generiert werden, wenn sich beide Nutzerprofile aufheben, also wenn sie sich genau ergänzen ($u_2 = -u_1$):

\begin{equation*}
u_{combined} = \frac{u_1}{||u_1||_2} + \frac{u_2}{||u_2||_2} = \frac{u_1}{||u_1||_2} + (-\frac{u_1}{||u_1||_2}) = 0
\end{equation*}

Dies ist beispielsweise der Fall wenn User 2 genau die Filme mag, die User 1 nicht mag und dieser die Filme positiv bewertet, die User 2 als negative Beispiele angegeben hat. In solchen Fällen ist es aber sowieso schwierig Filmempfehlungen zu erstellen und auch der erste Ansatz würde wohl keine brauchbaren Empfehlungen liefern.

Um die beiden Ansätze vergleichen zu können, wollen wir Top-10-Listen für kombinierte User erstellen und vergleichen. Wir starten mit einem einfachen Fall, bei dem beide Nutzenden gerne Harry-Potter-Filme sehen. Sie geben jeweils einen solchen aus dieser Reihe als positiv bewerteten Film an und daraus werden folgende Filmempfehlungen generiert:
"""

# %%
matrixBaseSbert = MatrixGenerator(
	metadata=True,
	genres=True,
	actors=True,
	directors=True,
	sbertEmbeddings='data/movies_sbert_5d.csv'
)
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
bestModel = Cakeommender("bestModel", config, matrixBaseSbert)

movies = pd.read_csv('./data/movies_meta.csv')
movies = movies[['movieId', 'original_title']].rename(
	columns={"original_title": "title"}).set_index('movieId')

def getRatingRows(movieTitles, userId, rating):
	ids = movies[movies.title.isin(movieTitles)].index
	if len(ids) != len(movieTitles):
		raise Exception('movie not found')

	return pd.DataFrame({'movieId': ids, 'userId': userId, 'rating': rating})

def compareStrategies(goodMovies1, badMovies1, goodMovies2, badMovies2, n=10, both=False):
	ratings_long = pd.concat((
		getRatingRows(goodMovies1, 1, 1),
		getRatingRows(badMovies1, 1, -1),
		getRatingRows(goodMovies2, 2, 1),
		getRatingRows(badMovies2, 2, -1)
	)).reset_index()

	ratings = ratings_long.pivot(index='movieId', columns='userId', values='rating')
	model = Cakeommender("compareModel", config, matrixBaseSbert, ratingsSplit=ratings)

	topN1Ids = model.intersectionTopNMoviesUser([1, 2], n).index
	topN1 = model.getMovieNamebyId(topN1Ids).fillna(topN1Ids.to_series())

	if both:
		topN2 = model.predictTopNForUser([1, 2], n).index
		topN2 = model.getMovieNamebyId(topN2).fillna(topN2.to_series())

	# display(Markdown('#### Filmempfehlungen'))
	lb = ', '
	display(Markdown(f"""
|             | **User 1** | **User 2** |
|---|------------|------------|
| **positiv** | {lb.join(goodMovies1)} | {lb.join(goodMovies2)} |
| **negativ** | {lb.join(badMovies1)} | {lb.join(badMovies2)}"""))

	if both:
		display(Markdown(f'Anzahl Überschneidungen: *{len(set(topN1) & set(topN2))}*'))
		topN1.index = range(1, len(topN1) + 1)
		topN2.index = range(1, len(topN2) + 1)

		display(Markdown('**Schnittmenge Empfehlungen**'))
		for index, title in enumerate(topN1):
			display(Markdown(f'{index + 1}. {title}'))

		display(Markdown('**Kombiniertes Nutzerprofil**'))
		for index, title in enumerate(topN2):
			display(Markdown(f'{index + 1}. {title}'))
	else:
		topN1 = topN1.to_frame()
		topN1['description'] = model.getMovieDescbyId(topN1Ids).fillna('')
		topN1.index = range(1, len(topN1) + 1)
		text = ''.join([f"""
{index}. **{record["title.1"]}**
{record.description}
			""" for index, record in topN1.iterrows()])
		display(Markdown(text))

compareStrategies(
	goodMovies1=['Harry Potter and the Deathly Hallows: Part 1'],
	badMovies1=[],
	goodMovies2=['Harry Potter and the Deathly Hallows: Part 2'],
	badMovies2=[],
	both=True
)

# %% [markdown]
"""
Es zeigt sich, dass die Filmempfehlungen für beide Varianten absolut identisch sind. Dies ist auch für andere Filmauswahlen und höhere N der Fall. In der Tat handelt es sich bei den beiden Varianten um dieselbe wie sich zeigen lässt:

In der ersten Variante berechnen wir die Cosine-Ähnlichkeit zwischen einem Film $m_i$ und den Nutzerprofilen $u_1$ und $u_2$ einzeln und berechnen daraus den Mittelwert, nachdem die Filme anschliessend auch sortiert werden. In der zweiten Variante berechnen wir die Ähnlichkeit anhand des kombinierten Nutzerprofils aus $u_1$ und $u_2$. Wenn beide Variante das gleiche Resultat liefern, müsste demnach folgende Aussage wahr sein:

\begin{equation*}
\frac{1}{2}\left(\cos(m_i, u_1) + \cos(m_i, u_2) \right) = \cos\left(m_i, \frac{u_1}{||u_1||} + \frac{u_2}{||u_2||}\right)
\end{equation*}
"""

# %% [markdown]
"""
Um dies zu zeigen, können wir nun einsetzen: $\cos(m, u) = \frac{m \cdot u}{||m|| \cdot ||u||}$

\begin{equation*}
\frac{1}{2}\left(\frac{m_i \cdot u_1}{||m_i|| \cdot ||u_1||} + \frac{m_i \cdot u_2}{||m_i|| \cdot ||u_2||} \right) = \frac{m_i \cdot (\frac{u_1}{||u_1||} + \frac{u_2}{||u_2||})}{||m_i|| \cdot ||\frac{u_1}{||u_1||} + \frac{u_2}{||u_2||}||}
\end{equation*}
"""

# %% [markdown]
"""
Da wir in der zweiten Variante die Nutzerprofile jeweils normieren, ergibt sich $||\frac{u_1}{||u_1||} + \frac{u_2}{||u_2||}|| = 2$

\begin{equation*}
\frac{1}{2}\left(\frac{m_i \cdot u_1}{||m_i|| \cdot ||u_1||} + \frac{m_i \cdot u_2}{||m_i|| \cdot ||u_2||} \right) = \frac{m_i \cdot (\frac{u_1}{||u_1||} + \frac{u_2}{||u_2||})}{2 \cdot ||m_i||}
\end{equation*}
"""

# %% [markdown]
"""
Nachdem wir die Terme umformen, können wir erkennen, dass die Aussage stimmt und somit beide Varianten tatsächlich identisch sind.

\begin{equation*}
\frac{1}{2}\left(\frac{m_i \cdot u_1}{||m_i|| \cdot ||u_1||} + \frac{m_i \cdot u_2}{||m_i|| \cdot ||u_2||} \right) = \frac{1}{2}\left(\frac{m_i \cdot u_1}{||m_i|| \cdot ||u_1||} + \frac{m_i \cdot u_2}{||m_i|| \cdot ||u_2||} \right)
\end{equation*}

Deswegen können wir darauf verzichten jeweils beide Varianten zu berechnen.

# Beispiele

Nachfolgend werden wir noch für unterschiedliche Nutzerpaare Filmempfehlungen des Recommendersystems ausgeben und deren Sinnhaftigkeit einschätzen.

## Filmempfehlungen für Comedy Liebhaber*innen
Ein erstes Beispiel besteht aus Nutzenden, die beide gerne Comedy-Filme sehen. Die eine Person mag den Film *Jumanji*, während die andere alle drei *Toy Story*-Filme positiv bewertet hat.

Die empfohlenen Filme passen alle ebenfalls ins Comedy-Genre und entsprechend gut zu den ausgewählten Filmen. Insbesondere *Toy Story Toons: Small Fry (2011)* ist ein sehr passender Film zu *Jumanji* und auch *Minions* und die Filme *Despicable Me* 2 und 3, die eine Weiterführung der *Minions* sind, passen gut zur *Toy-Story*-Reihe.
"""

# %%
compareStrategies(
	goodMovies1=['Jumanji: Welcome to the Jungle'],
	badMovies1=[],
	goodMovies2=['Toy Story 3', 'Toy Story of Terror!', 'Toy Story That Time Forgot'],
	badMovies2=[]
)

# %% [markdown]
"""
## Filmempfehlungen für Science-Fiction Liebhaber*innen mit Action Einfluss
Im nächsten Beispiel sehen wir uns die Empfehlungen für zwei Nutzende an, die beide Science-Fiction-Filme mögen. User 2 favorisiert dabei Action-Filme etwas stärker.

Da beide User den ersten *Avatar*-Film mögen, ist es konsequent, dass das Recommendersystem nun den zweiten, dritten und vierten Teil empfiehlt. Der Film *Rakka* passt zudem gut zu *Avatar* und *Ex Machina* und dürfte beiden gut gefallen. Die anderen Empfehlungen scheinen, aber eher ungeeignet zu sein.
"""

# %%
compareStrategies(
	goodMovies1=['Avatar: Creating the World of Pandora', 'Ex Machina'],
	badMovies1=[],
	goodMovies2=['Avatar: Creating the World of Pandora', 'Interstellar', 'Edge of Tomorrow'],
	badMovies2=[]
)

# %% [markdown]
"""
## Filmempfehlungen für Action Liebhaber*innen mit gegensätzlichen Präferenzen
Für das nächste Beispiel haben die Nutzenden jeweils auch Filme angegeben, die sich nicht mochten.
Der als negativ angegebene Film wird bei der jeweiligen anderen Person jedoch als Favorit definiert.
User 1 mag den Action-Film *Undisputed IV* und den Animationsfilm *Toy Story 3*, der bei User 2 nicht gut ankam.
Dieser wiederum mag *The Karate Kid* und *Undisputed III*, wobei letzterer beim ersten User schlecht abschneidet.
Beide scheinen aber Actionfilme zu mögen.

Das Recommendersystem empfiehlt nun *El Gringo*, was aufgrund der kurzen Beschreibung gut zur *Undisputed*-Reihe passen könnte. Es sind auch weitere Actionfilme in den Empfehlungen: *The Legend of Hercules*, *In the Name of the King III* und *Legendary: Tomb of the Dragon*. Bei letztgenanntem Film spielt zudem wie bei *Undisputed IV* der Schauspieler Scott Adkins mit. Was aber passenderweise nicht in den Vorschlägen vorkommt, sind Animationsfilme, Komödien und Kinderfilme, die zwar User 1 gefallen, denen aber User 2 abgeneigt ist.
"""

# %%
compareStrategies(
	goodMovies1=['Boyka: Undisputed IV', 'Toy Story 3'],
	badMovies1=['Undisputed III: Redemption'],
	goodMovies2=['The Karate Kid', 'Undisputed III: Redemption'],
	badMovies2=['Toy Story 3']
)
# %% [markdown]
"""
## Filmempfehlungen für Science-Fiction Liebhaber*innen mit gegensätzlichen Präferenzen
Beim folgenden Beispiel haben sich zwei Science-Fiction-Liebhabende getroffen.
User 1 gefällt die *Star Wars*-Reihe und hat bereits fast alle Filme davon gesehen.
Die *Star Trek*-Filme scheinen ihm nicht zu gefallen, von denen er zwei gesehen hat.
Diese gefallen aber User 2 sehr gut, der schon drei gesehen hat.

Die Empfehlungen enthalten einen der zwei *Star Wars*-Filme, die User 1 noch fehlen. *Star Wars: The Last Jedi* ist vorhanden, jedoch fehlt in den Vorschlägen *Rogue One: A Star Wars Story*, obwohl dieser passend wäre. Der *Star Trek*-Film *For the Love of Spock* fehlt korrekterweise in der Liste. Dies wäre zwar der letzte aus der Reihe, den User 2 noch nicht gesehen hat, aber da User 1 zwei der Reihe überhaupt nicht gefallen haben, erscheint es uns sinnvoll, wenn dieser nicht vorkommt. Die anderen empfohlenen Filme sind bis auf *Let's Kill Ward's Wife* alle auch dem Science-Fiction-Genre zuzuordnen und passen entsprechend gut. Warum dieser Ausreisser in den Vorschlägen ist, ist etwas sonderhaft, da die Genres und auch die Beschreibung weder zu den *Star-Wars*- noch *Star-Trek*-Filmen passen. Es gibt lediglich eine Übereinstimmung mit einem Schauspieler. Donald Faison spielt auch in *Robot Chicken: Star Wars Episode III* mit.
"""

# %%
compareStrategies(
	goodMovies1=['Solo: A Star Wars Story', 'Star Wars: The Force Awakens', 'Plastic Galaxy: The Story of Star Wars Toys', 'Robot Chicken: Star Wars Episode III'],
	badMovies1=['Star Trek Into Darkness', 'Star Trek: The Captains'],
	goodMovies2=['Star Trek Into Darkness', 'Star Trek: The Captains', 'Star Trek Beyond'],
	badMovies2=[]
)

# %% [markdown]
"""
# Fazit und nächste Schritte

Die Filmempfehlungen für unterschiedliche Nutzerprofile scheinen in den meisten Fällen sinnvoll zu sein. Es gibt zwar immer einzelne Vorschläge, die nicht zu den Eingaben passen, was aber nicht zwingend heissen muss, dass sie den Nutzenden nicht gefallen werden. Möglicherweise hat das Modell anhand der Filmbeschreibungen Ähnlichkeiten gefunden, die nicht direkt ersichtlich sind. Einzelne Empfehlungen, die nicht dem gewohnten entsprechen, können die Nutzenden auch dazu bewegen ihre Bubble zu verlassen und wieder mal etwas neues kennenlernen. Und sollten sie wirklich nicht glücklich mit solchen ungewöhnlichen Empfehlungen werden, können sie weiterhin aus den restlichen Vorschlägen auswählen, bei denen in der Regel immer ein Film dabei sein sollte, der für beide passen sollte.

Anhand des Prototyps konnten wir das Recommendersystem einfach testen und es auch von Freund*innen ausprobieren lassen. So erhielten wir auch Feedback zu den Vorschlägen, indem sie ihre eigenen Filmpräferenzen ausgewählt und dafür Vorschläge erhalten haben.

Den Prototyp könnte man nun weiterentwickeln und in eine richtige App umwandeln. Darin könnte man auch weitere Funktionen integrieren, damit diese wirklich einsetzbar wäre. Dazu gehören Features wie Registrierung bzw. Login, dem Speichern seiner Filmpräferenzen und das Durchstöbern der resultierenden Filmempfehlungen im Tinder-Stil.
"""

# %%
