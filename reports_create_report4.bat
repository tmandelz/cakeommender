
pipenv run jupytext --to notebook reports/report4.py
pipenv run nbmerge reports/report4.ipynb -o report4.ipynb
pipenv run jupyter nbconvert --execute --no-input report4.ipynb --to pdf --output report4.pdf