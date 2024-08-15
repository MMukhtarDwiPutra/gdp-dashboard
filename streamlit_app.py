import pandas as pd
import pickle
from PIL import Image
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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

programs = list(data["PROG"].value_counts().keys().sort_values())
selected_programs = st.sidebar.multiselect(label="Program", options=programs)

outputs_databar = data[(data["DATE"] >= start_date) &
               (data["DATE"] <= end_date)]

if selected_categories:
    outputs_databar = outputs_databar[outputs_databar['KATEGORI'].isin(selected_categories)]

if selected_presenter:
    outputs_databar = outputs_databar[outputs_databar['PRESENTER'].isin(selected_presenter)]

if selected_programs:
    outputs_databar = outputs_databar[outputs_databar['PROG'].isin(selected_programs)]

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

with open("data/used_data.pickle", "rb") as f:
    df = pickle.load(f)

###Predict by time
# Prepare the feature set and target
X = df[['START_MINUTES', 'END_MINUTES', 'DURASI']]
y_presenter = df['PRESENTER']  # Target 1
y_kategori = df['KATEGORI']    # Target 2

# Split the data into training and testing sets
X_train, X_test, y_presenter_train, y_presenter_test = train_test_split(X, y_presenter, test_size=0.3, random_state=42)
X_train, X_test, y_kategori_train, y_kategori_test = train_test_split(X, y_kategori, test_size=0.3, random_state=42)

# Initialize the models
model_presenter = RandomForestClassifier(random_state=42)
model_kategori = RandomForestClassifier(random_state=42)

# Train the models
model_presenter.fit(X_train, y_presenter_train)
model_kategori.fit(X_train, y_kategori_train)

# Predict on the full test set (to evaluate the model)
kategori_pred = model_kategori.predict(X_test)
presenter_pred = model_presenter.predict(X_test)

# Calculate accuracy on the full test set
accuracy_kategori = accuracy_score(y_kategori_test, kategori_pred)
accuracy_presenter = accuracy_score(y_presenter_test, presenter_pred)

st.header("Predict with Random Forest Classification")
st.text(f"Akurasi kategori {accuracy_kategori * 100:.2f}% \t" f"Akurasi presenter {accuracy_presenter * 100:.2f}%")

# Input form for time range
time_range_input = st.text_input("Predict with time range (e.g., 17:58 - 18:06)")

if time_range_input:
    start_time, end_time = time_range_input.split(' - ')
    start_minutes = int(pd.to_datetime(start_time, format='%H:%M').hour * 60 + pd.to_datetime(start_time, format='%H:%M').minute)
    end_minutes = int(pd.to_datetime(end_time, format='%H:%M').hour * 60 + pd.to_datetime(end_time, format='%H:%M').minute)
    duration = (pd.to_datetime(end_time, format='%H:%M') - pd.to_datetime(start_time, format='%H:%M')).seconds / 60
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'START_MINUTES': [start_minutes],
        'END_MINUTES': [end_minutes],
        'DURASI': [duration]
    })
    
    # Predict using the trained models
    presenter_pred = model_presenter.predict(input_data)
    kategori_pred = model_kategori.predict(input_data)

    # Display the results
    st.write(f"Predicted Presenter: {presenter_pred[0]}")
    st.write(f"Predicted Kategori: {kategori_pred[0]}")
