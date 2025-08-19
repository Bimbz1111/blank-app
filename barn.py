import streamlit as st

import pandas as pd

import plotly.express as px

import joblib

import numpy as np

st.title("Urban Sustainability Clustering")

st.header("Predict city clustering")

### Divide your screen into two columns
col1, col2 = st.columns(2)

#column 1 has some of the scaled features.
#st.number-input means the streamlit is fixed to numbers, it means you cam only write #numbers, you cant write words or alphabets
with col1:
    green_space = st. number_input("Green Space %", value=30.0)
    air_quality = st.number_input ("Air Quality Index", value=50.0)
    waste_recycled = st. number_input ("Waste Recycled %", value=50.0)
    renewable_energy = st.number_input ("Renewable Energy %", value=30.0)
    carbon_emissions = st. number_input ("darban Emissions", value=5.0)


## column 2 carries the rest of the features or columns
with col2:
    energy_efficiency = st.number_input ("Energy Efficiency", value=70.0)
    avg_commute = st. number_input ("Avg Commute (min)", value=35.0)
    water_access = st.number_input("Water Access %", value=95.0)
    population = st. number_input ("Population", value=5000000)


country = st.selectbox("Select Country for Map",
                      ["Germany", "United Kingdom", "France", "Spain", "canada",
                       "Australia", "New Zealand" , "United States",
                       "India", "Thailand", "Turkey", "Nigeria", "Egypt", "Kenya",
                       "Indonesia", "Pakistan", "Saudi Arabia", "United Arab Emirates", 
                       "Qatar", "Peru", "Denmark", "Sweden", "Finland", "Switzerland",
                       "Mexico", "South Africa", "Vietnam", "Philippines", "China",
                       "Russia", "Iran", "Iraq"])

if st.button("Predict Cluster"):


   model =joblib.load("kmeans_sustainability_model.joblib")
   scaler =joblib.load("scaler_sustainability.joblib")
   cluster_info= {0: "Transitional Cities", 1: "Critical Intervention Zone", 2: "Sustainability Leaders"}


input_data = np.array([[green_space, air_quality, waste_recycled, renewable_energy, 
                        carbon_emissions, energy_efficiency, avg_commute, water_access,
                        population]])

scale_input= scaler.transform(input_data)
#then we predict
prediction=model.predict(scale_input) [0]

st.success(f"predicted cluster is:{cluster_info[prediction]}")


#putting the map
#viz_data means the visualization data,and that pd.dataframe is the actual data we want to show,
viz_data = pd. DataFrame({
      'country': [country],
      'cluster': [cluster_info[prediction]]
  })
    
fig = px.choropleth(
viz_data,
locationmode='country names', 
locations='country', color='cluster',
title='Gloabl Sustainability Map', 
color_discrete_map={'Critical Intervention Zone':'red',
                      'Transitional Cities':'yellow',
                      'Sustainability Leaders': 'green'
                      },
hover_name='cluster'
  )

st.plotly_chart(fig, use_container_width=True)
# we need to tell stream lit to show the map, we use st.plotly isntead of fig show becos streamlit support plotly use container to be true so it does go out of span of the chosen arc


