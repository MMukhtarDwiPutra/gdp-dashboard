import pandas as pd
import pickle
from PIL import Image
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import plotly.tools as tls

st.markdown(
    """
    <style>
    /* Change sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #ADD8E6;
    }

    /* Optional: Adjust the text color in the sidebar */
    [data-testid="stSidebar"] .css-1d391kg {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
) 

with st.sidebar:
    st.page_link('streamlit_app.py', label='Introduction')
    st.page_link('pages/2_⌨️_Projects.py', label='⌨️ Projects')
    st.page_link('pages/3_📞_Contact.py', label='📞 Contact')


px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

with open("data/used_data.pickle", "rb") as f:
    data = pickle.load(f)


#Visualisasi dengan bar chart
st.header(":video_camera: TRENDING TOP 10 TOPIC THIS WEEK")

min_date = data["DATE"].min()
max_date = data["DATE"].max()
start_date, end_date = st.date_input(label="Rentang Waktu", 
                                             min_value=min_date,
                                             max_value=max_date,
                                             value=[min_date, max_date])

categories = list(data["KATEGORI"].value_counts().keys().sort_values())
selected_categories = st.multiselect(label="Kategori", options=categories)

presenters = list(data["PRESENTER"].value_counts().keys().sort_values())
selected_presenter = st.multiselect(label="Presenter", options=presenters)

programs = list(data["PROG"].value_counts().keys().sort_values())
selected_programs = st.multiselect(label="Program", options=programs)

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

# st.header(":bulb: Scatter Plot")
# col1, col2 = st.columns(2)
# # choice_1 = col1.selectbox('Horizontal', options=categories)
# # choice_2 = col2.selectbox('Vertical', options=presenters)
# fig_1 = px.scatter(data, 
#                  x=data["KATEGORI"], 
#                  y=data["PRESENTER"], 
#                  size="RATING", 
#                  hover_name="PRESENTER", 
#                  hover_data=["RATING"],
#                 #  title=f'Engagement dari {choice_1.title()} dan {choice_2.title()}'
#                  )

# fig_2 = px.scatter(data, 
#                  x=data["KATEGORI"], 
#                  y=data["PRESENTER"], 
#                  size="SHARE", 
#                  hover_name="PRESENTER", 
#                  hover_data=["SHARE"],
#                 #  title=f'Engagement dari {choice_1.title()} dan {choice_2.title()}'
#                  )

# st.plotly_chart(fig_1)
# st.plotly_chart(fig_2)

with open("data/used_data.pickle", "rb") as f:
    df = pickle.load(f)

###Predict by time
# Prepare the feature set and target
st.header("Best machine learning model for predict: Random Forest Regressor")
st.text("Prediksi dengan feature 'START_MINUTES', 'DURASI', 'KATEGORI', 'PRESENTER'")

# Initialize LabelEncoder
label_encoder_kategori = LabelEncoder()
label_encoder_presenter = LabelEncoder()

# Fit and transform the string data to integers
df['KATEGORI_NUMBER'] = label_encoder_kategori.fit_transform(df['KATEGORI'])
df['PRESENTER_NUMBER'] = label_encoder_presenter.fit_transform(df['PRESENTER'])

X = df[['START_MINUTES', 'DURASI', 'KATEGORI_NUMBER', 'PRESENTER_NUMBER']]
y_share = df['SHARE']  # Target 1
y_rating = df['RATING']    # Target 2

# Split the data
X_train_share, X_test_share, y_share_train, y_share_test = train_test_split(X, y_share, test_size=0.3, random_state=42)
X_train_rating, X_test_rating, y_rating_train, y_rating_test = train_test_split(X, y_rating, test_size=0.3, random_state=42)

forest_model_rating = RandomForestRegressor(random_state=42)
forest_model_share = RandomForestRegressor(random_state=42)

scaler_rating = StandardScaler()
scaler_share = StandardScaler()
X_train_share_scaled = scaler_share.fit_transform(X_train_share)
X_test_share_scaled = scaler_share.transform(X_test_share)
X_train_rating_scaled = scaler_rating.fit_transform(X_train_rating)
X_test_rating_scaled = scaler_rating.transform(X_test_rating)

# Train the models
forest_model_rating.fit(X_train_rating_scaled, y_rating_train)
forest_model_share.fit(X_train_share_scaled, y_share_train)

# Predict on the full test set (to evaluate the model)
rating_pred = forest_model_rating.predict(X_test_rating_scaled)
share_pred = forest_model_share.predict(X_test_share_scaled)

# Calculate mse on the full test set
mse = mean_squared_error(y_share_test, share_pred)
rmse_share = np.sqrt(mse)
r2_score_share = forest_model_share.score(X_test_share_scaled, y_share_test)

mse = mean_squared_error(y_rating_test, rating_pred)
rmse_rating = np.sqrt(mse)
r2_score_rating = forest_model_rating.score(X_test_rating_scaled, y_rating_test)

st.text(f"rmse rating Score: {rmse_rating :.2f}")
st.text(f"r2 rating Score: {r2_score_rating :.2f}")
st.text(f"rmse share Score: {rmse_share :.2f}")
st.text(f"r2 share Score: {r2_score_share :.2f}")

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_rating_test, rating_pred, color='blue', label='Predictions')
plt.plot([min(y_rating_test), max(y_rating_test)], [min(y_rating_test), max(y_rating_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('RandomForestRegressor Model: Actual vs Predicted Ratings')
plt.legend()
plt.grid(True)

# Convert the matplotlib figure to a plotly figure
plotly_fig = tls.mpl_to_plotly(plt.gcf())

# Display the plotly figure in Streamlit
st.plotly_chart(plotly_fig)

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_share_test, share_pred, color='blue', label='Predictions')
plt.plot([min(y_share_test), max(y_share_test)], [min(y_share_test), max(y_share_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Share')
plt.ylabel('Predicted Share')
plt.title('RandomForestRegressor Model: Actual vs Predicted Share')
plt.legend()
plt.grid(True)

# Convert the matplotlib figure to a plotly figure
plotly_fig = tls.mpl_to_plotly(plt.gcf())

# Display the plotly figure in Streamlit
st.plotly_chart(plotly_fig)

# Input form for time range
time_range_input = st.text_input("Predict with time range (e.g., 17:58 - 18:06)")

if time_range_input:
    start_time, end_time = time_range_input.split(' - ')
    start_minutes = int(pd.to_datetime(start_time, format='%H:%M').hour * 60 + pd.to_datetime(start_time, format='%H:%M').minute)
    end_minutes = int(pd.to_datetime(end_time, format='%H:%M').hour * 60 + pd.to_datetime(end_time, format='%H:%M').minute)
    durasi = (pd.to_datetime(end_time, format='%H:%M') - pd.to_datetime(start_time, format='%H:%M')).seconds / 60

    search_df = df[(df["START_MINUTES"] >= start_minutes) & (df["START_MINUTES"] <= end_minutes)]
    unique_presenter = search_df['PRESENTER'].unique()
    unique_kategori = search_df["KATEGORI"].unique()

    arr_result = []
    scaler = StandardScaler()

    st.header("Prediksi dengan feature 'START_MINUTES', 'DURASI', 'KATEGORI', 'PRESENTER'")
    for presenter in unique_presenter:
        for kategori in unique_kategori:
            kategori_encoded = label_encoder_kategori.transform([kategori])
            presenter_encoded = label_encoder_presenter.transform([presenter])

            input_data = pd.DataFrame({
                'START_MINUTES': start_minutes,
                'END_MINUTES': durasi,
                'KATEGORI_NUMBER': kategori_encoded,
                'PRESENTER_NUMBER': presenter_encoded,
            })
            
            input_data_rating = scaler_rating.fit_transform(input_data)
            input_data_share = scaler_share.fit_transform(input_data)
            
            predicted_rating = forest_model_rating.predict(input_data_rating)
            predicted_share = forest_model_share.predict(input_data_share)
            
            arr_result.append([kategori, presenter, predicted_rating[0], predicted_share[0]])

    data = np.array(arr_result)
    # Find the row with the maximum value in column 3 (index 2)
    max_value_rating = data[data[:, 2].astype(float).argmax()]
    max_value_share = data[data[:, 3].astype(float).argmax()]

    st.text("Prediksi rating tertinggi :")
    st.text(f"Kategori : {max_value_rating[0]}")
    st.text(f"Presenter : {max_value_rating[1]}")
    st.text(f"Rating : {max_value_rating[2]}")
    st.text(f"Share : {max_value_rating[3]}")
    st.text("")

    st.text("Prediksi share tertinggi :")
    st.text(f"Kategori : {max_value_share[0]}")
    st.text(f"Presenter : {max_value_share[1]}")
    st.text(f"Rating : {max_value_share[2]}")
    st.text(f"Share : {max_value_share[3]}")
