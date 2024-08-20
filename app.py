# %%
from dash import dcc, html, Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

# %%
import numpy as np
import pandas as pd
from libraries.pipeconfig import (
    CakeConfig,
    FeatureNormalizationEnum,
    SimilarityEnum,
    RatingScaleEnum,
)

from libraries.matrix import MatrixGenerator
from cakeommender import Cakeommender
# %%
movies = pd.read_csv(r'./data/movies_meta.csv')
movies = movies[['movieId', 'original_title']].rename(
    columns={"original_title": "title"})
# %%
app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.SKETCHY])
matrixBaseSbert = MatrixGenerator(
    metadata=True, genres=True, actors=True, directors=True, sbertEmbeddings='data/movies_sbert_5d.csv')
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


app.layout = html.Div(children=[
    dbc.Row(
        html.H1(children='Cakeommender')),

    dbc.Row(
        html.H3(children='Unsere App für den besten Filmabend zu Zweit'), style={
            'margin-top': '15px',
            'margin-left': '15px',
            'margin-right': '15px'
        }),
    dbc.Row(
        html.P(children='Damit die App funktioniert müssen kombiniert mindestens 3 Filme, die euch gefallen, gewählt werden!'), style={
            'margin-bottom': '25px',
            'margin-left': '15px',
            'margin-right': '15px'
        }),

    dbc.Row([
        dbc.Col(children=[
            html.Div(children='Welche Filme gefallen Dir?'),
            dcc.Dropdown(
                movies.title, placeholder="Wähle deine Lieblingsfilme", id='good_ones1', multi=True)
        ], width=6, style={
            'margin-left': '15px',
            'margin-right': '15px'}),
        dbc.Col(children=[
            html.Div(children='Welche Filme gefallen Dir?'),
            dcc.Dropdown(
                movies.title, placeholder="Wähle deine Lieblingsfilme", id='good_ones2', multi=True)
        ], style={
            'margin-left': '15px',
            'margin-right': '15px'})
    ], style={
        'margin-bottom': '25px',
    }),

    dbc.Row([
        dbc.Col(children=[
            html.Div(children='Welche Filme gefallen Dir überhaupt nicht?'),
            dcc.Dropdown(
                movies.title, placeholder="Bloss nicht diesen Film....", id='bad_ones1', multi=True)
        ], width=6, style={
            'margin-left': '15px',
            'margin-right': '15px'}),
        dbc.Col(children=[
            html.Div(children='Welche Filme gefallen Dir überhaupt nicht?'),
            dcc.Dropdown(
                movies.title, placeholder="Bloss nicht diesen Film....", id='bad_ones2', multi=True)
        ], style={
            'margin-left': '15px',
            'margin-right': '15px',
            'margin-bottom': '30px'})
    ]),

    dbc.Row(html.Div(
        html.Button('Los gehts', id='submit-val', n_clicks=0),
        style={'text-align': 'center',
               'margin-bottom': '30px'})),

    dbc.Row([
        dbc.Card([
            dbc.CardImg(src="/assets/Cake1.jpg", top=True),
            dbc.CardBody(class_name="card-cake-one", children=[
                html.Div(id="rec-cake-one-mv-title",
                         children=[html.H4("Cake", className="cake-one")]),
                html.Div(id="rec-cake-one-mv-desc", children=[html.P(
                    "Text of film description",
                    className="card-text")]),
            ])
        ], style={"width": "17rem",
                  'margin-left': '15px',
                  'margin-right': '15px'}),
        dbc.Card([
            dbc.CardImg(src="/assets/Cake2.jpg", top=True),
            dbc.CardBody(children=[
                html.Div(id="rec-cake-two-mv-title",
                         children=[html.H4("Cake", className="cake-two")]),
                html.Div(id="rec-cake-two-mv-desc", children=[html.P(
                    "Text of film description",
                    className="card-text")]),
            ])
        ], style={"width": "17rem",
                  'margin-left': '15px',
                  'margin-right': '15px'}),
        dbc.Card([
            dbc.CardImg(src="/assets/Cake3.jpg", top=True),
            dbc.CardBody(children=[
                html.Div(id="rec-cake-three-mv-title",
                         children=[html.H4("Cake", className="cake-three")]),
                html.Div(id="rec-cake-three-mv-desc", children=[html.P(
                    "Text of film description",
                    className="card-text")]),
            ])
        ], style={"width": "17rem",
                  'margin-left': '15px',
                  'margin-right': '15px'}),
        dbc.Card([
            dbc.CardImg(src="/assets/Cake4.png", top=True),
            dbc.CardBody(children=[
                html.Div(id="rec-cake-four-mv-title",
                         children=[html.H4("Cake", className="cake-four")]),
                html.Div(id="rec-cake-four-mv-desc", children=[html.P(
                    "Text of film description",
                    className="card-text")]),
            ])
        ], style={"width": "17rem",
                  'margin-left': '15px',
                  'margin-right': '15px'}),
        dbc.Card([
            dbc.CardImg(src="/assets/Cake5.jpg", top=True),
            dbc.CardBody(children=[
                html.Div(id="rec-cake-five-mv-title",
                         children=[html.H4("Cake", className="cake-five")]),
                html.Div(id="rec-cake-five-mv-desc", children=[html.P(
                    "Text of film description",
                    className="card-text")]),
            ])
        ], style={"width": "17rem",
                  'margin-left': '15px',
                  'margin-right': '15px'})
    ], style={'margin-left': '15px',
              'margin-right': '15px'}, justify="center")
])

@app.callback(Output('rec-cake-one-mv-title', 'children'), Output('rec-cake-one-mv-desc', 'children'),
              Output('rec-cake-two-mv-title', 'children'), Output('rec-cake-two-mv-desc', 'children'),
              Output('rec-cake-three-mv-title','children'), Output('rec-cake-three-mv-desc', 'children'),
              Output('rec-cake-four-mv-title', 'children'), Output('rec-cake-four-mv-desc', 'children'),
              Output('rec-cake-five-mv-title', 'children'), Output('rec-cake-five-mv-desc', 'children'),
               [Input('submit-val', 'n_clicks'), Input('good_ones1', 'value'), 
               Input('good_ones2', 'value'), Input('bad_ones1', 'value'), Input('bad_ones2', 'value')])
def on_click(n_clicks, goodMovies_user1, goodMovies_user2, badMovies_user1, badMovies_user2):
    if goodMovies_user1 is not None and goodMovies_user2 is not None and len(goodMovies_user1) + len(goodMovies_user2) >= 3:
        
        goodMovies_user1_movieId = bestModel.getMovieIdbyName(goodMovies_user1)
        goodMovies_user2_movieId = bestModel.getMovieIdbyName(goodMovies_user2)

        if badMovies_user1 is not None and badMovies_user2 is not None:
            badMovies_user1_movieId = bestModel.getMovieIdbyName(badMovies_user1)
            badMovies_user2_movieId = bestModel.getMovieIdbyName(badMovies_user2)

            bestModel.calcAppUserProfiles(
                [goodMovies_user1_movieId, goodMovies_user2_movieId])
            userProfileGoodMovies = bestModel.userProfiles
            bestModel.calcAppUserProfiles(
                [badMovies_user1_movieId, badMovies_user2_movieId])
            bestModel.userProfiles = -1 * bestModel.userProfiles + userProfileGoodMovies

        else:
            bestModel.calcAppUserProfiles(
                [goodMovies_user1_movieId, goodMovies_user2_movieId])

        recommendations = bestModel.predictTopNForUser(
            users=["0", "1"], n=20, removeRatedMovies=False)
        if badMovies_user1 is not None and badMovies_user2 is not None:
            combinedIdRec = goodMovies_user1_movieId.tolist() + goodMovies_user2_movieId.tolist() + badMovies_user1_movieId.tolist() + badMovies_user2_movieId.tolist()
        else:
            combinedIdRec = goodMovies_user1_movieId.tolist() + goodMovies_user2_movieId.tolist()

        recommendationId = np.array(recommendations.index)[[id not in combinedIdRec for id in recommendations.index]][:5]
        movienames = bestModel.getMovieNamebyId(recommendationId).reset_index(drop=True)
        moviedescs = bestModel.getMovieDescbyId(recommendationId).reset_index(drop=True)

        return movienames[0], moviedescs[0], movienames[1], moviedescs[1], movienames[2], moviedescs[2], movienames[3], moviedescs[3], movienames[4], moviedescs[4]

    else:
        return ["", "", "", "", "", "", "", "", "", ""]


if __name__ == '__main__':
    app.run_server(debug=True)

# %%
