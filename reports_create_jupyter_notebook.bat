pipenv run jupytext --to notebook reports/prereport3.py
pipenv run jupytext --to notebook modelling/random_cakeommender.py
pipenv run jupytext --to notebook modelling/base_cakeommender.py
pipenv run jupytext --to notebook modelling/tfidf_cakeommender.py
pipenv run jupytext --to notebook modelling/bert_cakeommender.py
pipenv run jupytext --to notebook modelling/sbert_cakeommender.py
pipenv run jupytext --to notebook modelling/best_combined_cakeommender.py


pipenv run nbmerge reports/prereport3.ipynb modelling/random_cakeommender.ipynb modelling/base_cakeommender.ipynb modelling/tfidf_cakeommender.ipynb modelling/bert_cakeommender.ipynb modelling/sbert_cakeommender.ipynb modelling/best_combined_cakeommender.ipynb -o report3.ipynb


pipenv run jupytext --to notebook reports/prereport2.py
pipenv run jupytext --to notebook tools/dataretrieval.py
pipenv run jupytext --to notebook eda/eda_movies.py
pipenv run jupytext --to notebook eda/eda_crew_casts.py
pipenv run jupytext --to notebook eda/ratings.py
pipenv run nbmerge reports/prereport2.ipynb tools/dataretrieval.ipynb eda/eda_movies.ipynb eda/eda_crew_casts.ipynb eda/ratings.ipynb -o report2.ipynb