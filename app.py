
################################################################################################################################
################################################ Importation des bibliothéques #################################################
################################################################################################################################

import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels

import seaborn as sns # Visualization
import matplotlib.pyplot as plt # Visualization
#from colorama import Fore

from matplotlib import pyplot
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose


from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

from datetime import datetime,date,timedelta 
import datetime as dt
import time

import warnings # Supress warnings 
warnings.filterwarnings('ignore')

np.random.seed(7)
import IPython
from IPython.display import display
import base64

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
import hydroeval as he


import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io 
import requests as rq
from io import BytesIO




py.init_notebook_mode(connected=True)




################################################################################################################################
########################################### hide streamlit components ##########################################################
################################################################################################################################

#hide_menu_style = """ 
#        <style> 
#        #MainMenu {visibility : hidden; }
#        footer {visibility : hidden;}
#        </style>
#        """
#st.markdown(hide_menu_style, unsafe_allow_html = True)




################################################################################################################################
################################################ barre latérale ################################################################
################################################################################################################################


with st.sidebar:
    choose = option_menu("Senegal River Forecasting 📉📈", ["Accueil", "Lstm", "Article", "Code source"],
                         icons=['house', 'clipboard-data', 'clipboard-data', 'clipboard-data'],
                         menu_icon="app-indicator",default_index=0, 
                         styles={                  
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#3379FF"},
    }
    )
    
    



################################################################################################################################
################################################ page d'accueil ################################################################
################################################################################################################################

if choose == "Accueil":
        
        
        
        # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Accueil</p>', unsafe_allow_html=True)        
        st.markdown("""
         Cette application a été conçue dans le but de prédire l'évolution du régime hydrolique du fleuve Sénégal.
         Elle a été réalisée grâce à une variante des réseaux de neuronnes de récurrents. Il s'agit des réseau récurrent 
         à mémoire court et long terme ou plus explicitement les réseaux de neurones récurrents à mémoire court-terme 
         et long terme (LSTM: Long Short Term Memory).""")
        
        

        

        
            
        
            

    
    
    
################################################################################################################################
################################################Lstm univarié ##################################################################
################################################################################################################################    

elif choose == "Lstm":
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Lstm</p>', unsafe_allow_html=True)
     
    #with st.expander(label="Veuillez cliquer pour déplier/replier"):
        

    station = ('','BAKEL', 'MATAM', 'PODOR')
    selected_station = st.selectbox('Veuillez cliquer pour sélectionner une station', station, key =0)

    ##############################################################################################
    ## Traitement des données brutes
    ##############################################################################################

    if selected_station =='':

        st.markdown('')
    else:
        #

        #n_years = st.slider('Années prévisonnelles:', 1, 4)
        #period = n_years * 365

        ### Chargment des données
        @st.cache
        def readerfile():


            if selected_station ==  'BAKEL':
                url = "https://github.com/AYLY92/memoire/blob/main/data/H_Bakel.xls"
                data = rq.get(url).content
                data = pd.read_excel(BytesIO(data), nrows = 366)

            elif selected_station ==  'MATAM':
                url = "https://github.com/AYLY92/memoire/blob/main/data/H_Matam.xls"
                data = rq.get(url).content
                data = pd.read_excel(BytesIO(data), nrows = 366).drop('Unnamed: 49', 1)

            else:
                url = "https://github.com/AYLY92/memoire/blob/main/data/H_Podor.xls"
                data = rq.get(url).content
                data = pd.read_excel(BytesIO(data), nrows = 366).drop('Unnamed: 49', 1)                     
                                     

            #data.reset_index(inplace=True)

            return data



        def configure_plotly_browser_state():

            import IPython
            display(IPython.core.display.HTML('''
                  <script src ="/static/components/requirejs/require.js"></script>
                  <script>
                    requirejs.config({
                      paths: {
                        base:'/static/base',
                        plotly:'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',

                      },
                    ));
                    </script>
                    '''))

        #Télécharment des fichiers sous format csv
        #def filedownload(df):
    
        #    csv = df.to_csv(index=False)
        #    b64 = base64.b64encode(csv.encode()).decode()
        #    href = f'<a href="data:file/csv;base64,{b64}" download ="station.csv">Télécharger le fichier CSV</a>'
        #    return href
        @st.cache
        def convert_df(df):
             # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        

        data_load_state = st.text('Loading data...')
        data = readerfile()
        data_load_state.text('Loading data... done! ✔️')


        st.markdown('''
        ### Traitement des valeurs brutes
        ''')
        st.markdown('**Tableau des données brutes**')
        st.write('Dimensions des données: ' + str(data.shape[0]) + ' lignes & ' + str(data.shape[1]) + ' colonnes')
        st.write(data)
        #st.markdown(filedownload(data), unsafe_allow_html=True)
        csv = convert_df(data)
        st.download_button(
             label="Télécharger le tableau",
             data=csv,
             file_name='data.csv',
             mime='text/csv',
         )

        ### Prétraitement des données
        @st.cache
        def prepoocessing():
            if selected_station ==  'BAKEL':
              url = "https://github.com/AYLY92/memoire/blob/main/data/df_bakel.csv"                       
              data = pd.read_csv(url).drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
              if data['hauteur'].isna().sum() != 0:
                data['hauteur'] = data['hauteur'].interpolate()
              if data['hauteur'].isna().sum() != 0:
                data['hauteur'] = data['hauteur'].fillna(df['hauteur'].mean())

            elif selected_station ==  'MATAM':
              url="https://github.com/AYLY92/memoire/blob/main/data/df_matam.csv"                       
              data = pd.read_csv(url).drop(['Unnamed: 0'], axis=1)
              if data['hauteur'].isna().sum() != 0:
                data['hauteur'] = data['hauteur'].interpolate()
              if data['hauteur'].isna().sum() != 0:
                data['hauteur'] = data['hauteur'].fillna(df['hauteur'].mean())

            else:
              url="https://github.com/AYLY92/memoire/blob/main/data/df_podor.csv"                       
              data = pd.read_csv(url).drop(['Unnamed: 0'], axis=1)
              if data['hauteur'].isna().sum() != 0:
                data['hauteur'] = data['hauteur'].interpolate()
              if data['hauteur'].isna().sum() != 0:
                data['hauteur'] = data['hauteur'].fillna(data['hauteur'].mean())

            #data.reset_index(inplace=True)

            return data




        data_load_state = st.text('Loading data...')
        data = prepoocessing()
        data_load_state.text('Loading data... done! ✔️')

        st.markdown('**Tableau des données prétraitées**')
        st.write('Dimensions des données: ' + str(data.shape[0]) + ' lignes & ' + str(data.shape[1]) + ' colonnes')
        st.write(data)
        csv = convert_df(data)
        st.download_button(
             label="Télécharger le tableau",
             data=csv,
             file_name='data.csv',
             mime='text/csv',
         )
        #st.markdown(filedownload(data), unsafe_allow_html=True)




        ### Visualisation
        def plot_raw_data():

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['date'], y=data['hauteur'].fillna(method='ffill')))
            fig.layout.update(xaxis_rangeslider_visible=True, xaxis =dict(title = 'date'), yaxis = dict(title = "hauteur d'eau"))
            st.plotly_chart(fig)

        st.markdown("**Visualisation de l'évolution du régime hydrologique**")	
        plot_raw_data()




        ## Traitement des données prédites
        # renvoie un dateframe constitué de la colonne date et chacune des données de la station et les dimensions de décompositions
        def creation_df(df, col_date, col_y, n):

            train_size = int(n * len(df))
            test_size = len(df) - train_size

            univariate_df = df[[col_date, col_y]].copy()
            return univariate_df, train_size, test_size


        univariate_df, train_size, test_size = creation_df(data, 'date', 'hauteur', 0.85)


        # Normalisation et division en données de train et de test
        from sklearn.preprocessing import MinMaxScaler

        def normalize_data(df, col_y):
            # Normalisation des données
            data = df.filter([col_y])
            dataset = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)


            return scaled_data, scaler



        scaled_data, scaler = normalize_data(univariate_df, 'hauteur')



        # Split into train et test for scaled_data
        def split_scaled_data(scaled_data, train_size, look_back):

            train, test = scaled_data[:train_size-look_back,:], scaled_data[train_size-look_back:,:]
            return train, test



        train, test = split_scaled_data(scaled_data, train_size, 365)


        # Création du dataset qui convertit un tableau de valeurs en une matrice de données.
        def create_dataset(dataset, look_back):
            X, Y = [], []
            for i in range(look_back, len(dataset)):
                a = dataset[i-look_back:i, 0]
                X.append(a)
                Y.append(dataset[i, 0])
            return np.array(X), np.array(Y)

        #Bakel
        x_train, y_train = create_dataset(train, 365)
        x_test, y_test = create_dataset(test, 365)


        # remodeler l'entrée pour qu'elle soit [échantillons, pas de temps, caractéristiques].
        def reshaping(x_train, x_test):
            return np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])), np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

        x_train, x_test = reshaping(x_train, x_test)

        #load the save model
        if selected_station ==  'BAKEL':
            model = tf.keras.models.load_model('./models/models_lstm/best_model_bakel.h5')

        elif selected_station ==  'MATAM':
            model = tf.keras.models.load_model('./models/models_lstm/best_model_matam.h5')

        else:
            model = tf.keras.models.load_model('./models/models_lstm/best_model_podor.h5')

        # Prédiction
        def make_prediction(model, x_train, x_test, y_train, y_test, scaler):
            # Lets predict with the model
            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)

            # invert predictions
            train_predict = scaler.inverse_transform(train_predict)
            y_train = scaler.inverse_transform([y_train])

            test_predict = scaler.inverse_transform(test_predict)
            y_test = scaler.inverse_transform([y_test])

            # Get the root mean squared error (RMSE) and MAE on test data
            #score_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
            score_nse = he.evaluator(he.nse, list(y_test[0]), list(test_predict[:,0]))[0]
            score_mae = mean_absolute_error(y_test[0], test_predict[:,0])
            #print('RMSE: {}'.format(score_rmse))
            print('MAE: {}'.format(score_mae))
            print('Nash: {}'.format(score_nse))

            return score_nse, score_mae, y_test, test_predict

        score_nse, score_mae, y_test, test_predict = make_prediction(model, x_train, x_test, y_train, y_test, scaler)


        # Tableau comparatif des valeurs prédites vs valeurs réelles
        def tab_prediction(df, test_size, y_test, test_predict):
            x_test_ticks = df.tail(test_size)['date']
            y = y_test[0]
            y_pred = test_predict[:,0]
            return pd.DataFrame({'date': x_test_ticks, 'Test Set': y, 'Predict on Test Set': y_pred})

        #Test Set
        st.markdown('**Tableau Test Set vs Prediction on Test Set**')
        tab_test = tab_prediction(univariate_df, test_size, y_test, test_predict)
        st.write('Dimensions des données: ' + str(tab_test.shape[0]) + ' lignes & ' + str(tab_test.shape[1]) + ' colonnes')
        st.write(tab_test)
        csv = convert_df(tab_test)
        st.download_button(
             label="Télécharger le tableau",
             data=csv,
             file_name='tab_test.csv',
             mime='text/csv',
         )
        #st.markdown(filedownload(tab_test), unsafe_allow_html=True)
        st.write('MAE  : ' + str(score_mae)) 
        st.write('NASH : ' + str(score_nse))



        # Visualisation de la prédiction
        def visualisation_prediction(df, train_size, test_size, y_test, test_predict):

            x_train_ticks = df.head(train_size)['date']
            y_train = df.head(train_size)['hauteur']
            x_test_ticks = df.tail(test_size)['date']

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_train_ticks, y=y_train, name='Train Set'))
            fig.add_trace(go.Scatter(x=x_test_ticks, y=test_predict[:,0],  name= "Prediction on Test Set"))
            fig.add_trace(go.Scatter(x=x_test_ticks, y=y_test[0], name = "Test Set"))
            fig.layout.update( {"title":"Hauteur d'eau prédite vs réelle",
                      "xaxis":{"title": 'date',
                               "zeroline":False},
                      "yaxis":{"title":'hauteur',
                               "zeroline":False}} )
            st.plotly_chart(fig) 

        st.markdown("**Visualisation des valeurs prédites sur les données d'entrainement et de test**")
        visualisation_prediction(univariate_df, train_size, test_size,  y_test, test_predict)

        #  future prediction
        def futur_prediction(model, test_data, look_back,  nday):

            from numpy import array
            x_input = scaler.inverse_transform(test_data[len(test_data) - look_back:].reshape(1, -1))
            temp_input=list(x_input)
            temp_input=temp_input[0].tolist()

            lst_output=[]
            #look_back=365
            i=0
            while(i<nday):

                if(len(temp_input)>look_back):
                    #print(temp_input)
                    x_input=np.array(temp_input[1:])
                    #print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1, -1)
                    x_input = x_input.reshape((1, 1, look_back))
                    #print(x_input)
                    yhat = model.predict(x_input, verbose=0)
                    yhat = scaler.inverse_transform(yhat)
                    #print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]
                    #print(temp_input)
                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1,  1, look_back))
                    yhat = model.predict(x_input, verbose=0)
                    yhat = scaler.inverse_transform(yhat)
                    #print(yhat[0])
                    temp_input.extend(yhat[0].tolist())
                    #print(len(temp_input))
                    lst_output.extend(yhat.tolist())
                    i=i+1

            lst_output = pd.DataFrame(lst_output, columns= ['hauteur'])
            day_pred =  pd.DataFrame(pd.date_range('2008-05-01', periods=nday, freq='1D').tolist(), columns=['date'])
            pred_station = pd.concat([day_pred, pd.DataFrame(lst_output).iloc[:,0]], 1)

            return pred_station

        ##############################################################################################
        ## Traitement des valeurs prédites
        ##############################################################################################
        st.markdown('''
        ### Traitement des valeurs predites
        ''')
        # nombre d'année à prédire
        period = st.number_input(
        label="Insérer le nombre de jours à prédire",
        min_value=0,
        step=1, key=0)

        if period == 0:

            st.markdown('')

        else:

            data_load_state = st.text('Loading data...')
            nday = period
            pred_station = futur_prediction(model, test, 365, nday)
            data_load_state.text('Loading data... done! ✔️')

            st.markdown("**Tableau des valeurs prédictes**")
            st.write('Dimensions des données: ' + str(pred_station.shape[0]) + ' lignes & ' + str(pred_station.shape[1]) + ' colonnes')
            st.write(pred_station)
            csv = convert_df(pred_station)
            st.download_button(
                 label="Télécharger le tableau",
                 data=csv,
                 file_name='pred_station.csv',
                 mime='text/csv',
             )
            #st.markdown(filedownload(pred_station), unsafe_allow_html=True)


            # visualisation_future_prediction

            def visualisation_future_prediction(univariate_df, pred_station, look_back, nday):


                new_df = pd.concat([univariate_df, pred_station]).reset_index(drop=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=new_df.iloc[len(univariate_df) - look_back:, 0], y=new_df.iloc[len(univariate_df) - look_back:, 1],  name= "Hauteur d'eau réelle"))
                fig.add_trace(go.Scatter(x=new_df.iloc[len(univariate_df):,0], y= new_df.iloc[len(univariate_df):,1], name = "Hauteur d'eau prévue"))
                fig.layout.update(title="Hauteur d'eau prévue dans "+str(nday)+ " jours", title_x=0.5, 
                  xaxis=dict(
                      rangeselector=dict(
                          buttons=list([
                              dict(count=1,
                                  label='1m',
                                  step='month',
                                  stepmode='backward'),
                              dict(count=6,
                                  label='6m',
                                  step='month',
                                  stepmode='backward'),
                              dict(count=12,
                                  label='1y',
                                  step='month',
                                  stepmode='backward'),
                              dict(count=36,
                                  label='3y',
                                  step='month',
                                  stepmode='backward'),
                              dict(step='all')
                          ])
                      ),
                      rangeslider=dict(
                          visible = True
                      ),
                      title='date'
                  ),
                  yaxis = dict(title="Hauteur d'eau")
              )
                st.plotly_chart(fig)

            st.markdown("**Visualisation des valeurs prédites**")
            visualisation_future_prediction(univariate_df, pred_station, 365, nday)
    
    
            

    
    
    
################################################################################################################################
################################################ ARTICLE ################################################################
################################################################################################################################    

elif choose == "Article":
            
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Article</p>', unsafe_allow_html=True)
    #with st.expander(label="Veuillez cliquer pour déplier/replier"):
        

    st.markdown("""
        Veuilez cliquer sur le lien ci-dessous pour accéder à l'artilce
         
        * **Article:** [https://github.com](https://github.com/AYLY92/memoire/tree/main/Rapport)
        
        """)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    

    
    

    
################################################################################################################################
################################################ CODE SOURCE ##########################################################################
################################################################################################################################

elif choose == "Code source":
            
    st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Code source</p>', unsafe_allow_html=True)
    st.markdown("""
         Veuillez cliquer sur le lien ci-dessous pour accéder au code source.

        * **Code source:** [https://github.com](https://github.com/AYLY92/memoire/tree/main/code%20source)
        """)
    
     
    
