import pandas as pd
import pickle
from PIL import Image
import streamlit as st
import os

if st.button("Logout"):
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.write('<meta http-equiv="refresh" content="0;url=/">', unsafe_allow_html=True)
else:
    with st.sidebar:
        st.page_link('streamlit_app.py', label='Introduction')
        st.page_link('pages/2_⌨️_Projects.py', label='⌨️ Projects')
        st.page_link('pages/4_⌗_Data.py', label='⌗ Data')
        st.page_link('pages/3_📞_Contact.py', label='📞 Contact')
        
    # Ensure the "Data" folder exists
    if not os.path.exists('data'):
        os.makedirs('data')
    st.header("Input Data")

    uploaded_files = st.file_uploader(
        "Choose an Excel file", accept_multiple_files=True, type=['xlsx']
    )

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        
        # Save the uploaded file to the "Data" folder
        with open(os.path.join("data", "Dataset.xlsx"), "wb") as f:
            f.write(bytes_data)
        
        st.write(f"File 'Dataset.xlsx' saved to 'Data' folder.")

    st.header("Download Data")

    with open("data/Dataset.xlsx", "rb") as file:
        btn = st.download_button(
            label="Download Excel",
            data=file,
            file_name="Dataset.xlsx",
            mime="xlsx",
        )
