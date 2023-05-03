"""----------------------------------------------------------------------------
    utils.py

    Laurent VOURIOT

    last update : 27/02/2022
-------------------------------------------------------------------------------"""

import random 
import base64
from io import BytesIO

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

from matplotlib.figure import Figure
import matplotlib.pyplot as plt, mpld3

from plotly.utils import PlotlyJSONEncoder
import plotly.express as px
import json
import re

def get_correspondence_map(df):
    correspondence_map = {}
    for col in df.columns:
        categorical = df[col].astype('category').cat.codes
        correspondence_map[col] = {nominal : categorical for nominal, categorical in zip(df[col], categorical)}
    return correspondence_map

# -----------------------------------------------------------------------------

df_categories = pd.read_excel('./categories.xlsx')
correspondence_map = get_correspondence_map(df_categories)
antibiotiques = df_categories['Antibiotique'].dropna()
df_categories = df_categories.drop(columns='Antibiotique')

# -----------------------------------------------------------------------------

def debug_msg(msg):
    """Readable debug msg to print in flask console. """
    return '\n\033[93m*************************************\n \
            {} \
            \n*************************************\033[0m\n'.format(msg)

# -----------------------------------------------------------------------------

def get_code(value, field):
    """
    Get numerical code for categorical value.

    -----------------------------------
    args :
    value : (str) categorical value (e.g 'Salmonella typhimurium')
    field : (str) variable name (e.g 'Espece')
    -----------------------------------
    returns : (int) code associated to the value of the given field.
    """
    if value == '--Pas d\'informations--':    
        return -1
    return correspondence_map[field][value] 
    
# -----------------------------------------------------------------------------

def get_model_predict(dataframe):
    """
    Predict the resistance of all the antibiotics given the values selected by the 
    user.
    
    Depending on the selected values for each field the stade is not the same so 
    we have to call a different model. He are the needed variables for each stade.
    
    stade 1 : date, bmr, service, prelevement
    stade 2 : date, bmr, service, prelevement, direct
    stade 3 : date, bmr, service, prelevement, culture
    stade 4 : date, bmr, service, prelevement, culture, espece
    
    -----------------------------------
    args :
    dataframe : (pd.DataFrame) dataframe containing the variables selected by the user.
    -----------------------------------
    returns : (list(float)) probabilities of sensitivity for each antibiotic.
    """
    
    if None not in dataframe[['Date', 'BMR_ATCD', 'Service', 'Prelevement', 'Culture', 'Espece']].values[0]:
        dataframe = dataframe.drop('Direct', axis='columns')
        stade = 4
    elif dataframe['Espece'].values[0] is None  \
            and None not in dataframe[['Date', 'BMR_ATCD', 'Service', 'Prelevement', 'Culture']].values[0]:
        dataframe = dataframe.drop(['Espece', 'Direct'], axis='columns')
        stade = 3
    elif dataframe[['Espece','Culture']].values[0].all() is None and \
            None not in dataframe[['Date', 'BMR_ATCD', 'Service', 'Prelevement', 'Direct']].values[0]:
        dataframe = dataframe.drop(['Espece','Culture'], axis='columns')
        stade = 2
    elif dataframe[['Espece', 'Culture','Direct']].values[0].all() is None and\
            None not in dataframe[['Date', 'BMR_ATCD', 'Service', 'Prelevement']].values[0]:
        dataframe = dataframe.drop(['Direct', 'Espece', 'Culture'], axis='columns')
        stade = 1
    else:
        raise ValueError
                            
    model = load_model('./saved_models_global/NNTotal/'+str(stade)+'Keras.h5', compile=False)
    return model.predict(dataframe)[0]

# -----------------------------------------------------------------------------

def render_plot(spectres_antibiotique, pred_dict):
    """
    Generates the plot of the probability sensitivity (y axis) and the spectrum (x axis) of 
    each antibiotic. 

    -----------------------------------
    args : 
    spectres_antibiotique : (pd.DataFrame) 
    y_scores : (list(float)) probability of sensitivity.
    -----------------------------------
    returns : (str) base64 encoding of the plot to be able to display it in the web app. 
    """
    df_spectres = pd.read_excel('Resultats_Spectre.xlsx')
    df_spectres['Antibiotique'] = df_spectres['Antibiotique'].apply(lambda x : re.sub(r'/','-', str(x)))
    
    df_pred = pd.DataFrame(columns=['Antibiotique', 'Prédiction'])
    df_pred['Antibiotique'] = pred_dict.keys()
    df_pred['Prédiction'] = pred_dict.values()

    df_results = df_spectres.merge(df_pred, on='Antibiotique')
    
    fig = px.scatter(df_results, x='Spectre1', y='Prédiction', hover_name='Antibiotique')

    graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)
    return graphJSON

    """
    buffer = io.StringIO()

    df = px.data.iris() # replace with your own data source
    fig = px.scatter(
        df, x="sepal_width", y="sepal_length", 
        color="species")
    fig.write_html(buffer)

    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    return encoded
    
    fig = Figure(figsize=(3,2))
    ax = fig.subplots()

    ax.scatter(df_spectres['Spectre1'], y_scores)
    ax.set_xlabel('spectre')
    ax.set_ylabel('probabilité de sensibilité')
   #  ax.axvline(max(df_spectres['Spectre1'])/2, linestyle='--', color='black', lw=1)
   #  ax.axhline(max(y_scores)/2, linestyle='--', color='black', lw=1)
    
    # TODO dynamic plot

    for i, label in enumerate(antibiotiques):
        if y_scores[i] > 0.5 and df_spectres['Spectre1'][i] < max(df_spectres['Spectre1'])/2:
            ax.annotate(label, (df_spectres['Spectre1'][i], y_scores[i]), rotation=15, ha='left')
    
    return mpld3.fig_to_html(fig)
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data
    """
