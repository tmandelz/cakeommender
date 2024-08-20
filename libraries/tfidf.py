import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""
This library is used for generating the recommender's feature matrix using the TF-IDF algorithm.
"""

def tokenize(data: pd.DataFrame, stopword: bool, lowercase: bool, minDf = 1, maxDf = 1.0, ngramRange = (1, 1)) -> pd.DataFrame:
    """
        TF-IDF to tokenize movie overviews (https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

        :param pd.DataFrame data: Data Frame with the movie overview
        :param bool stopword: True to delete the stopwords
        :param bool lowercase: True to change all words to lowercase
        :param int mind_df: minimum number of appearance
        :param float max_df: maximum number of appearance between 0 and 1
        :param tuple ngram_range: range of the ngram

        :return: Returns the tokens with movieId as index
        :rtype: pd.DataFrame
    """

    try:
        if stopword:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngramRange, lowercase=lowercase, min_df=minDf, max_df=maxDf)
        else:
            vectorizer = TfidfVectorizer(ngram_range=ngramRange, lowercase=lowercase, min_df=minDf, max_df=maxDf)

        vectors = vectorizer.fit_transform(data["overview"])
        nlpFeatures = pd.DataFrame(vectors.todense(), columns=vectorizer.get_feature_names_out())
        df_tokens = nlpFeatures.set_index(data.movieId)
    except Exception as e:
        print(f"Exception with tokenizing overview column: \n {e}")
        raise e

    return df_tokens
