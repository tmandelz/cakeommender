![](/assets/app_overview.PNG)
# Cakeommender
This project is a part of the [CDS1 Cakeommender Group](https://gitlab.fhnw.ch/joseph.weibel/cakeommender) at [Data Science FHNW](https://www.fhnw.ch/en/degree-programmes/engineering/bsc-data-science).

#### -- Project Status: Completed

## Project Intro/Objective
Before we buy a new product, we usually ask our friends, research the product features, compare the product with similar products and read product reviews on the Internet. How convenient would it be if all these processes were automatic and the product was recommended efficiently?

In this challenge, you have the opportunity to implement a product recommendation system yourself, which can efficiently make personalized product recommendations based on product descriptions. 

Content-based recommender systems are particularly advantageous for start-up companies with few customers or for serving new customers or customers who rarely return. These recommend products based on product characteristics in text or images and their similarity between individual products, while also taking personal preferences into account.

### Methods Used
* Modelling
* Content based Recommender System
* Hybrid Recommender Systems
* Distance and Similarity Calculations
* Dashboard development


### Technologies
* Python
* Latex
* Pandas
* HTML
* CSS
* CSV

## Overview Folder Structure
* Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
* Raw Data is being kept [here](data)
* Original Movielense Data is being kept [here](movielens_data)
* Modelling Scripts for misc. Recommenders are being kept [here](modelling)
* Libraries and Functions for Recommenders are being kept [here](libraries)
* Scripts for EDA are being kept [here](eda)
* CSS and other assets for the App are being kept [here](assets)
* Assets for report generation are being kept [here](reports)
* Misc Tools are being kept [here](tools)

## Featured Files
* [A Demo File for the Recommender](demobaseline.py)
* [The Main Recommender File](cakeommender.py)
* [The Main Evaluation File](evaluation.py)
* [The Main Webapp File](app.py)


## Executing the Dash Web-app
- Run `app.py` in your IDE or run `python .\app.py` from your CLI
- Call `localhost:8050` in your Browser

## Installation Pipenv Environment
### Voraussetzungen
- Pipenv installed in local Python Environment [Pipenv](https://pipenv.pypa.io/en/latest/) or just run `pip install pipenv` in your CLI
### First Installation of Pipenv Environment
- open your CLI
- run `cd /your/local/github/repofolder/`
- run `pipenv install`
- Restart VS Code or IDE
- Choose the newly created "cakeommender" Virtual Environment python Interpreter

### Environment already installed (Update dependecies)
- open your CLI
- run `cd /your/local/github/repofolder/`
- run `pipenv sync`

## Instantiating a Basic Recommender
An example of a base and hybrid (base + SBERT) recommender can be found in the [Demo File](demobaseline.py).

To run the recommender and to get an example of Top-N Movies the following instructions can be used.
- open your CLI
- run `python .\demobaseline.py`


## Contributing Members
**[Daniela Herzig](https://github.com/dcherzig)**
**[Thomas Mandelz](https://github.com/tmandelz)**
**[Joseph Weibel](https://gitlab.fhnw.ch/joseph.weibel/)**
**[Jan Zwicky](https://github.com/swiggy123)**
