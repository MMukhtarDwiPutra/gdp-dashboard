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

categories = list(data["KATEGORI"].value_counts().keys().sort_values())
selected_categories = st.sidebar.multiselect(label="Kategori", options=categories)

presenters = list(data["PRESENTER"].value_counts().keys().sort_values())
selected_presenter = st.sidebar.multiselect(label="Presenter", options=presenters)

outputs_databar = data[(data["DATE"] >= start_date) &
               (data["DATE"] <= end_date)]

if selected_categories:
    outputs_databar = outputs_databar[outputs_databar['KATEGORI'].isin(selected_categories)]

if selected_presenter:
    outputs_databar = outputs_databar[outputs_databar['PRESENTER'].isin(selected_presenter)]

avg_rating = outputs_databar["RATING"].mean()
avg_share = outputs_databar["SHARE"].mean()

#Visualisasi dengan bar chart
st.header(":video_camera: TOPIK")
st.text(f"Average rating: {avg_rating:.2f}\t" f" Average share: {avg_share:.2f}")

bar_data_1 = outputs_databar.groupby("CG TOPIK")["RATING"].sum().nlargest(10).sort_values(ascending=True)
fig_1 = px.bar(bar_data_1, x="RATING", color=bar_data_1, orientation='h')
st.plotly_chart(fig_1)

#Visualisasi dengan bar chart
bar_data_2 = outputs_databar.groupby("CG TOPIK")["SHARE"].sum().nlargest(10).sort_values(ascending=True)
fig_2 = px.bar(bar_data_2, x="SHARE", color=bar_data_2, orientation='h')
st.plotly_chart(fig_2)

#Visualisasi dengan scatter plot
outputs_scatter = data[["KATEGORI", "PRESENTER"]]

st.header(":bulb: Engagement")
col1, col2 = st.columns(2)
# choice_1 = col1.selectbox('Horizontal', options=categories)
# choice_2 = col2.selectbox('Vertical', options=presenters)
fig_1 = px.scatter(data, 
                 x=data["KATEGORI"], 
                 y=data["PRESENTER"], 
                 size="RATING", 
                 hover_name="PRESENTER", 
                 hover_data=["RATING"],
                #  title=f'Engagement dari {choice_1.title()} dan {choice_2.title()}'
                 )

# fig_2 = px.scatter(data, 
#                  x=data["KATEGORI"], 
#                  y=data["PRESENTER"], 
#                  size="SHARE", 
#                  hover_name="PRESENTER", 
#                  hover_data=["SHARE"],
#                 #  title=f'Engagement dari {choice_1.title()} dan {choice_2.title()}'
#                  )

st.plotly_chart(fig_1)
# st.plotly_chart(fig_2)
