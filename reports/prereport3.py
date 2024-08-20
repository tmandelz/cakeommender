#%% [markdown]
# \begin{titlepage}
#    \begin{center}
#       \vspace*{1cm}
#       \Huge
#       \textbf{Report 3}

#       \vspace{0.5cm}
#       Tinder for Movies - Challenge HS 2022

#       \vspace{1.5cm}

#       \Large
#       Daniela Herzig, Thomas Mandelz, Joseph Weibel, Jan Zwicky

#       \normalsize
#       Dezember 2022

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

# %% [markdown]
"""

Dieser Bericht beschreibt die Modellierung der Recommendersysteme und deren Evaluierung.

# Pipeline
Am Anfang der Aufgabenstellung wollten wir gleich mit der Modellierung beginnen. Aus vorherigen Challenges und Projekten wussten wir, dass ein sauberer Aufbau der Pipeline uns auf die Dauer Zeit ersparen würde. Aufgrund dessen haben wir die Anforderungen an ein Recommendersystem definiert, welche uns im weiteren Rahmen der Challenge hilft, unterschiedliche Modelle und deren Konfigurationen zu testen und zu evaluieren.

<br><br>
<center>

![cakeommender](reports/res/OverviewCakeommender.png)

</center>
<br><br>

Das Flussdiagram soll einen vereinfachten Ablauf des Recommendationprozesses für das Recommendersystem darstellen.
Beim Codieren der Pipeline hielten wir uns an diesen vorab definierten Recommendationprozess und konnten effizient die konfigurierbaren Elemente umsetzen.


## Namensgebung
Unsere Pipeline haben wir `cakeommender` genannt. Der Name hat nur einen losen Zusammenhang mit den Filmen, welche er vorschlagen soll.
An einem der ersten Meetings haben wir als Gedankenstütze zusammen einen Kuchen gegessen.
Inspiriert durch den Kuchen haben wir dem Recommendersystem den "provisorischen" Namen `cakeommender` gegeben.
Dieser Name besteht nun aber bis zum Ende der Challenge!

## Konfiguration
Wir möchten mit möglichst wenig Codezeilen ein Recommendersystem definieren und konfigurieren können.
Dazu benötigen wir ein Konfigurationsobjekt, das der Pipeline übergeben werden kann. Anhand von diesem soll das Recommendersystem korrekt initialisiert und erstellt werden.

### CakeConfig
Mithilfe eines `CakeConfig` Objekts kann die gesamte Konfigurierung erstellt werden. Sie besteht aus den folgenden Attributen:
weightsDistribution, similarity, ratingScale und featureNormalisation

<br>
<br>
**weightsDistribution:**
Die Weights-Distribution ist ein Zeilenvektor von Dezimalzahlen (0-inf), dessen Dimensionen der Länge der Features entspricht. Dieser Vektor wird verwendet um die Features zu gewichten.
<br>
<br>
**similarity:**
Die Similarity besteht aus einem Enum (Kategorie), welche die folgenden Werte annehmen kann: `Cosine`, `Pearson` und `Jaccard`
Der übergebene Wert bestimmt das Ähnlichkeitsmass, welches das Recommendersystem zur Berechnung der Userprofil-Filmprofil-Ähnlichkeiten verwendet. Jaccard wurde bei unserer Modellierung nicht eingesetzt, da es für unsere Features ein ungeeignetes Mass ist.
<br>
<br>
**ratingScale:**
Die ratingScale besteht aus einem Enum (Kategorie), welche die folgenden Werte annehmen kann: `BINARY`, `TERTIARY` und `STANDARDISED`
Der übergebene Wert bestimmt die Skala, in der die Ratings dem Recommendersystem zur Verfügung gestellt wird.
<br>
<br>
**featureNormalisation:**
Die featureNormalisation besteht aus einem Enum (Kategorie), das die folgenden Werte annehmen kann: `NONE`, `MINMAX` (zwischen 0 und 1), `ZSCORE` (standardisiert)
<br>

Der übergebene Wert bestimmt somit die Art der Normalisierung, in welcher die Features im Recommendersystem normalisiert werden. Im Falle von `NONE` werden die unveränderten Features übernommen.

## WeightGenerator
Der `WeightGenerator` definiert die Gewichte der einzelnen Features. Zu Beginn wollten wir alle Features einzeln gewichten. Wir haben aber schnell gemerkt, dass die Granularität viel zu fein und diese auch nur begrenzt wirksam wäre.
Die Gewichtung kann  auf die einzelnen Featuregruppen (bsp. Genres, Metadaten, TFIDF Tokens, Embeddings, etc.) gemacht werden.

## MatrixGenerator
Der `MatrixGenerator` ist eine von uns entwickelte Klasse, um effizient Features und Ratings von verschiedenen CSV-Quellen einzulesen.
Mittels den verschiedenen Enums für die Features oder Ratingskalen kann ein Subset (bsp. nur Genres) an Daten als Matrix geladen werden.
Da wir für die Evaluation das Recommendersystem mehrfach intialisieren müssen und dies anfangs relativ zeitintensiv war, haben wir einige Verbesserungen vorgenommen.
Anstatt die CSV-Datei normal mit pandas einzulesen verwenden wir konstant die `pyarrow`-Engine für pandas. Deren einzige Nachteile sind ein schlechteres Handling von "Zeit- und Datumswerten", die aber in unseren Datensets nicht vorkommen.
Eine weitere Verbesserung ist ein Caching der bereits eingelesenen Feauture oder Ratingsmatrizen. Durch diese Optimierungen lassen sich Recommendersysteme in kurzer Zeit erstellen.

## Normalisation
Die Features können mit verschiedenen Varianten normalisiert werden.

## Userprofiles
Nachdem die Features und Ratings korrekt eingelesen wurden, werden alle Userprofile berechnet.

<br>
\begin{equation*}
u = \text{number of users}
\quad m = \text{number of movies}
\quad f = \text{number of features}
\end{equation*}

<br>

\begin{equation*}
\text{Ratings} =
\begin{pmatrix} r_{11} & r_{12} & \cdots & r_{1u}\\
r_{21} & r_{22} & \cdots & r_{2u}\\
\vdots & \vdots & \ddots & \vdots\\
r_{m1} & r_{m2} & \cdots & r_{mu}
\end{pmatrix}
\end{equation*}



\begin{equation*}
\text{Features} =
\begin{pmatrix}
f_{11} & f_{12} & \cdots & f_{1f}\\
f_{21} & f_{22} & \cdots & f_{2f}\\
\vdots & \vdots & \ddots & \vdots\\
f_{m1} & f_{m2} & \cdots & f_{mf}
\end{pmatrix}
\end{equation*}
Mithilfe einer Matrix mit Ratings und einer Matrix mit Features wird eine Userprofil-Matrix berechnet.
<br>
Dazu müssen die Dimensionen beachtet werden. Das Recommendersystem garantiert schon beim Initialisieren, dass sowohl die Feature-Matrix als auch die Ratings-Matrix dieselben **movies** enthält.
Die Durchführbarkeit der Matrixmultplikation ist somit gewährleistet und wird folgendermassen ausgeführt:

\begin{equation*}\underset{u\times f}{\mathrm{Userprofiles}} = \underset{u\times m}{\mathrm{Ratings^T}} \times \underset{m\times f}{\mathrm{Features}}\end{equation*}
Die berechneten Userprofile werden im Recommendersystem gespeichert und können weiterverwendet werden für den eigentlichen Recommendationprozess.

## Movie-User Similarities
Um die besten Filme für einen User zu finden, müssen die Ähnlichkeiten der Filmprofile zu den Userprofilen gefunden werden.
Dazu wird die Ähnlichkeit der gesamten Featurematrix (Movieprofil) zu den Userprofilen berechnet.
Nachfolgend die Berechnung beispielshalber mit der Cosine Ähnlichkeit:

\begin{equation*}\underset{m\times u}{\mathrm{Similarities}} = Sim_{cos}( \underset{m\times f}{\mathrm{Features}} ,\underset{u\times f}{\mathrm{Userprofiles}})\end{equation*}


Diese neue komplette Similaritymatrix wird im Recommendersystem abgelegt und kann für das Zusammenstellen der Top-N Empfehlungen verwendet werden.

## Top-N-Empfehlungen für User
Die Top-N-Recommendations können aus den vorherigen Similarities ausgelesen werden.

### Einzeluser
Für einen User $u$ wurde die Ähnlichkeit seines Userprofils zu den Filmen berechnet.
Das heisst die Top-N-Liste ist die Spalte $u$ in der Similaritymatrix absteigend sortiert nach der grössten Ähnlichkeit.
\begin{equation*}\underset{m \times 1}{\mathrm{top_{u}}} = sort_{desc}(\underset{m\times 1}{\mathrm{Similarities_u}})\end{equation*}

Davon werden die obersten $n$ Film-IDs genommen und zurückgegeben.
<br>

### Combined Users
Für ein kombiniertes Userprofil benötigt es noch einen Zwischenschritt.
Die Userprofile werden jeweils durch ihre Norm geteilt, so dass sie Länge 1 besitzen. Dadurch erhalten beide das gleiche Gewicht wenn sie für das kombinierte Profil aufsummiert werden.
\begin{equation*}\underset{1\times f}{\mathrm{combinedUserprofile}} = \sum_{n=1}^{|users|} \frac{u_{n}} {|| u_{n} ||}\end{equation*}
Danach werden die Similarities aus dem Kapitel **Movie-User Similarities** für dieses kombinierte Profile berechnet.
Der restliche Prozess folgt dann dem Einzeluserablauf.


# Evaluierung

Sobald wir ein Recommendersystem konfiguriert und berechnet haben, können wir dieses evaluieren, um deren Qualität beurteilen und sie untereinander vergleichen zu können. Dazu haben wir eine Evaluation-Library entwickelt, die für alle Systeme die Metriken und Plots gleichermassen berechnet.

## Precision

Wir wollen unsere Modelle an der Precision messen, da wir an möglichst vielen guten Vorschlägen in den Top-N-Empfehlungen interessiert sind. Deswegen berechnen wir einerseits Precision@10 und andererseits auch die Precision anhand aller bekannten Ratings. Letzteres um alle User im Datenset für unsere Evaluierung zu berücksichtigen. Für Precision@n sind für jeden User mindestens $n$ positive Ratings und $n$ negative Ratings nötig, damit durch das Recommendersystem auch wirklich Werte zwischen 0 und 1 erreicht werden können. Zudem verlangen wir mindestens drei zusätzliche Ratings für die Berechnung der Empfehlungen. Diese drei Ratings sind notwendig, um ein Userprofil erstellen zu können. Für Precision@10 wären somit mindestens 23 Ratings pro User nötig. Viele User weisen aber weniger Ratings auf. Eine Lösung würde darin bestehen, nur User im Datenset zu belassen, die dieses Kriterium erfüllen, was aber zu einem Bias in der Evaluierung führen würde. Die Evaluierung würde User mit wenigen Ratings ignorieren und wir könnten nicht überprüfen, wie sich das Recommendersystem für User mit wenigen Ratings verhält.

<br>
Aus diesem Grund verkleinern wir das Datenset nicht weiter und berechnen stattdessen die Precision@10 für User mit der Mindestanzahl an Ratings und die Precision für alle User. Für die generelle Precision setzen wir zudem auf fünffache Kreuzvalidierung, um eine Abschätzung für den Fehler dieser Metrik zu erhalten. Deswegen können wir die Mindestanzahl an Ratings auf lediglich 5 senken. Beim Erstellen der Folds achten wir jeweils darauf, dass die Ratings pro User im Trainings- und Testteil gleich verteilt sind. Der Trainingsteil wird anschliessend für das Berechnen des Nutzerprofils verwendet.

\begin{equation*}\text{precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}\end{equation*}

## Novelty

Neben einer hohen Precision, möchten wir auch möglichst diverse Vorschläge in den Top-10-Empfehlungen haben. Wir möchten verhindern, dass die Recommendersysteme hauptsächlich Blockbusters vorschlagen, die zwar bei vielen Usern auf Gefallen stossen würden, was aber nicht die Aufgabe eines Recommendersystems mit persönlichen Empfehlungen sein sollte. Je höher dieser Wert ist, desto unbekanntere Filme werden vorgeschlagen. Der kleinste Wert beträgt 0. Diese Metrik eignet sich für den Vergleich unter den Recommendersystemen. Da sie ordinalskaliert ist, kann nicht genau beurteilt werden, wie stark sich ein Modell verbessert oder verschlechtert hat, wenn sich deren Wert erhöht oder verringert.

<br>
Diese Metrik berechnen wir anhand der Formel aus dem [Post im rsy-Space vom 02.05.2021](https://spaces.technik.fhnw.ch/spaces/recommender-systems/beitraege/recommender-system-evaluierung-coverage-und-novelty)

\begin{equation*}\text{novelty} = -\frac{1}{u} \sum^u_{i=1} \sum^{n}_{j=1} \frac{log_2(\text{popularity}(\text{top}_j))}{n}\end{equation*}

Die Popularity wird berechnet indem die Anzahl Ratings eines Films durch die Anzahl Filme geteilt wird.

## Qualitative Evaluierung

Neben der quantitativen Evaluierung führen wir während des Modellierungsprozesses laufend qualitative Evaluierungsmethoden durch. Einerseits untersuchen wir die Empfehlungen anhand der Genres im Vergleich zu den Genres in den Nutzerprofilen. Anderseits werden wir bei den Modellen mit  NLP-Ansätzen die Features genauer betrachten und versuchen darin Muster zu erkennen. Ausserdem werden wir Top-N-Listen von den erstellen Systemen abfragen und deren Qualität selbst subjektiv einschätzen.

# Vorgehen mit unterschiedlichen Recommendersystemen

Nach dieser theoretischen Einführung in unseren Prozess, beginnen wir mit der Modellierung. Wir starten mit einem Recommendersystem, das auf Zufallswerten als Features basiert. Die berechneten quantitativen Metriken können wir als Baselinewerte betrachten, die keines der folgenden Systeme unterschreiten sollte. Das System wird die Empfehlungen durch zufällig ausgewählte Filme zusammensetzen und jedes weitere System sollte eine intelligentere Strategie dafür aufweisen.

<br>
Um die Qualität der Modelle mit NLP-Features beurteilen zu können, erstellen wir zudem auch Recommendersysteme mit herkömmlichen Features wie Budget, Umsatz, Filmdauer, Genres, Schauspieler*innen und Regisseur*innen. Dabei testen wir auch unterschiedliche Konfigurationen aus und wollen ideale Parameter für die nachfolgenden Modelle bestimmen. Wir erwarten, dass die Metriken dieses Baseline-Systems durch die Systeme auf Basis der NLP-Features übertroffen werden.

<br>
Nach der Modellierung mit TFIDF-Tokens und BERT- und SBERT-Embeddings wollen wir die Modelle weiter verbessern, indem wir die unterschiedlichen Features miteinander kombinieren und optimal gewichten. Zuletzt wollen wir das beste Recommendersystem nochmals genauer betrachten und ein Fazit ziehen, bevor wir es später für unsere Anwendung verwenden.

"""

# %%
import os
if os.getcwd().endswith('reports'):
    os.chdir('..')
