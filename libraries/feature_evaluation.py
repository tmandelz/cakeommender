import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN

def plotSimiliarities(features: pd.DataFrame, feature_name: str, return_sim: bool = False) -> plt.figure:
	"""
	Joseph Weibel
	calculates cosine similarities between all rows in the given DataFrame and plots the results in a histogram.
	:param features pd.DataFrame: rows for which the similarities should be calculated
	:param feature_name str: name of the features in the DataFrame which is used for the title of the plot
	:param return_sim bool: if similarities should be returned along with the figure
	:return: plot showing similarities and similarities itself if requested
	:rtype: plt.figure, np.array
	"""
	sim = 1 - pairwise_distances(features, features, metric='cosine', n_jobs=-1)
	print('Minimum Ähnlichkeit:', sim.min())
	print('Maximum Ähnlichkeit:', sim.max())

	fig, ax = plt.subplots(figsize = (10, 5))
	ax.hist(sim.flatten(), bins=50, density=True)
	ax.set_title(f'Verteilung Ähnlichkeiten zwischen Filmprofilen ({feature_name}) ($\mu = {sim.mean().round(3)}$)')
	ax.set_xlabel('Ähnlichkeit')
	ax.set_ylabel('rel. Häufigkeit')

	if return_sim:
		return fig, sim
	else:
		return fig

def plotHDBSCANProbabilities(model: HDBSCAN) -> plt.figure:
	"""
	Joseph Weibel
	shows probability for belonging to a cluster for each sample in a histogram
	:param model HDBSCAN: fitted HDBSCAN models
	:return: plot showing probabilities
	:rtype: plt.figure
	"""
	fig, ax = plt.subplots(figsize = (10, 5))
	ax.hist(model.probabilities_, bins=50, density=True)
	ax.set_title('Verteilung Wahrscheinlichkeiten Cluster-Zugehörigkeit')
	ax.set_xlabel('Wahrscheinlichkeit')
	ax.set_ylabel('rel. Häufigkeit')

	return fig

def plotHDBSCANClustersAlongGenres(model: HDBSCAN, genres: pd.DataFrame) -> plt.figure:
	"""
	Joseph Weibel
	shows the number of movies in each cluster for each genres as a stacked barplot
	:param model HDBSCAN: fitted HDBSCAN models
	:param genres pd.DataFrame: genres to plot
	:return: plot showing probabilities
	:rtype: plt.figure
	"""
	clusters = genres.copy()
	clusters['cluster'] = model.labels_
	clusters = clusters.groupby('cluster').sum()

	fig, ax = plt.subplots(figsize = (10, 5))
	clusters.T.plot.bar(stacked=True, ax=ax)
	ax.set_title('Aufteilung Genres nach Cluster')
	ax.set_xlabel('Genre')
	ax.set_ylabel('Anteil pro Cluster')

	return fig

def plotUMAPEmbeddingsAlongGenres(umap_embeddings: np.array, genres: pd.DataFrame) -> plt.figure:
	"""
	Joseph Weibel
	shows a scatter plot for each genre with the two dimensional embeddings in it.
	:param umap_embeddings np.array: two dimensional array after UMAP reduction
	:param genres pd.DataFrame: genres to plot
	:return: plot showing probabilities
	:rtype: plt.figure
	"""
	fig, axs = plt.subplots(5, 4, figsize=(20, 30), sharex='all', sharey='all')
	axs = axs.flatten()
	fig.delaxes(axs[-1])

	for i, genre in enumerate(genres.columns):
		embeddings_of_genre = umap_embeddings[genres[genre] == 1]
		axs[i].scatter(embeddings_of_genre[:,0], embeddings_of_genre[:,1], s=0.5)
		axs[i].set_title(genre)

		axs[i].set_xlabel('Komponente 1')
		axs[i].set_ylabel('Komponente 2')

	return fig

def plotUMAPEmbeddingsGenreAlongTopNGenres(umap_embeddings: np.array, genres: pd.DataFrame, main_genre: str, n: int = 3):
	"""
	Joseph Weibel
	Shows the embeddings assigned to the given main genre in a scatter plot along with the top n genres assigned to these movies.
	For each top n genre a scatter plot with all movies assigned to it will be shown.
	Movies assigned to the main genre will be highlighted in them.
	:param umap_embeddings np.array: two dimensional array after UMAP reduction
	:param genres pd.DataFrame: genres used to get the assignment for each movie
	:param main_genre str: genre to highlight
	:param n int: number of other genres to show along main genre
	:return: plots with n + 1 subplots showing embeddings for movies with main genre and its top n other genres
	:rtype: plt.figure
	"""
	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

	fig, axs = plt.subplots(1, 1 + n, figsize=(5 * (n + 1), 6), sharex='all', sharey='all')
	axs = axs.flatten()
	axs[0].set_ylabel('Komponente 2')
	for ax in axs:
		ax.set_xlabel('Komponente 1')

	other_genres = genres[genres[main_genre] == 1]
	top_n_genres = other_genres.sum().sort_values(ascending=False).iloc[1:n+1].index
	embeddings_of_genre = umap_embeddings[genres[main_genre] == 1]

	cat = other_genres[top_n_genres].apply(lambda x: x[x == 1].index, axis=1)
	cat = cat.apply(lambda x: ', '.join(x) if len(x) <= 1 else 'multiple')
	cat = cat.replace('', 'others')

	palette_colors = sns.color_palette('tab10')
	genre_colors = dict(
		others='gray',
		multiple='yellow',
		**{genre: color for genre, color in zip(top_n_genres, palette_colors)}
	)

	sns.scatterplot(
		x=embeddings_of_genre[:,0],
		y=embeddings_of_genre[:,1],
		hue=cat,
		s=5,
		ax=axs[0],
		palette=genre_colors
	)
	axs[0].set_title(main_genre)
	# sns.move_legend(axs[0], "upper left", bbox_to_anchor=(0, -0.12), ncol=4)

	handles, labels = axs[0].get_legend_handles_labels()
	handles = [
		*[handles[labels.index(genre)] for genre in top_n_genres],
		handles[labels.index('multiple')],
		handles[labels.index('others')]
	]
	axs[0].legend(handles, [*top_n_genres, 'multiple', 'others'])

	for i, genre in enumerate(top_n_genres):
		other_genres = genres[genres[genre] == 1]
		cat = np.where(other_genres[main_genre] == 1, main_genre, 'others')
		s = np.where(other_genres[main_genre] == 1, 20, 5)
		embeddings_of_genre = umap_embeddings[genres[genre] == 1]
		sns.scatterplot(
			x=embeddings_of_genre[:,0],
			y=embeddings_of_genre[:,1],
			hue=cat,
			s=s,
			ax=axs[i+1],
			palette={main_genre: genre_colors[genre], 'others': genre_colors['others']}
		)
		axs[i + 1].set_title(genre)

		handles, labels = axs[i+1].get_legend_handles_labels()
		handles = [handles[labels.index(main_genre)], handles[labels.index('others')]]
		axs[i+1].legend(handles, [main_genre, 'others'])

	np.warnings.filterwarnings('default', category=np.VisibleDeprecationWarning)

	fig.suptitle(f'Genre {main_genre} und Top-{n} zugeordnete Genres')

	return fig

def plotCumulativePCAVariance(model: PCA, threshold: float = 0.95) -> plt.figure:
	"""
	Joseph Weibel
	Plots a CDF plot showing the cumulative variance in the given PCA model.
	Marks the number of components required to keep the given amount of variance in the data.

	:param model PCA: fitted PCA model for which the variance should be shown
	:param threshold float: relative cumulative variance between 0 and 1 which should be kept in the data and marked in the plot
	:return: plot showing probabilities and the number of componented required to reach the given variance threshold.
	:rtype: plt.figure, int
	"""
	cumsum_variance = model.explained_variance_ratio_.cumsum()
	n_components = np.argmax(cumsum_variance >= threshold) + 1

	fig, ax = plt.subplots(figsize = (10, 5))
	ax.grid()
	ax.plot(range(len(model.explained_variance_ratio_)), cumsum_variance)
	ax.vlines(x=n_components, ymin=cumsum_variance.min(), ymax=cumsum_variance.max() + 0.02, color='r')
	ax.scatter([n_components], [threshold], marker='o', s=200, edgecolors='r', c='#ffffff00', linewidths=1)
	ax.set_title('Kumulative erklärte Varianz')
	ax.set_xlabel('Anzahl Principal Components')
	ax.set_ylabel('erklärte Varianz')
	ax.set_ylim(cumsum_variance.min(), cumsum_variance.max() + 0.02)

	return fig, n_components
