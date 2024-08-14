import pandas as pd
import pickle
from PIL import Image
import streamlit as st
import plotly.express as px

px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

with open("data/used_data.pickle", "rb") as f:
    data = pickle.load(f)

min_date = data["DATE"].min()
max_date = data["DATE"].max()
start_date, end_date = st.sidebar.date_input(label="Rentang Waktu", 
                                             min_value=min_date,
                                             max_value=max_date,
                                             value=[min_date, max_date])

categories = ["Semua Kategori"] + list(data["KATEGORI"].value_counts().keys().sort_values())
category = st.sidebar.selectbox(label="Kategori", options=categories)

presenters = ["Semua Presenter"] + list(data["PRESENTER"].value_counts().keys().sort_values())
presenter = st.sidebar.selectbox(label="Presenter", options=presenters)

outputs = data[(data["DATE"] >= start_date) &
               (data["DATE"] <= end_date)]

if category != "Semua Kategori":
    outputs = outputs[outputs["KATEGORI"] == category]

if presenter != "Semua Presenter":
    outputs = outputs[outputs["PRESENTER"] == presenter]

#Visualisasi dengan bar chart
st.header(":video_camera: TOPIK")
bar_data_1 = outputs.groupby("CG TOPIK")["RATING"].sum().nlargest(10).sort_values(ascending=True)
fig_1 = px.bar(bar_data_1, x="RATING", color=bar_data_1, orientation='h')
st.plotly_chart(fig_1)

#Visualisasi dengan bar chart
bar_data_2 = outputs.groupby("CG TOPIK")["SHARE"].sum().nlargest(10).sort_values(ascending=True)
fig_2 = px.bar(bar_data_2, x="SHARE", color=bar_data_2, orientation='h')
st.plotly_chart(fig_2)
