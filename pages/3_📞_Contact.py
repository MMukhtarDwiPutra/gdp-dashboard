import streamlit as st
from PIL import Image

st.title("Contact Us")

# Define the contact information
contact_info = {
    "Phone": {
        "icon": "📞",
        "text": "(+654) 6544 55",
        "description": "Lorem ipsum dolor sit"
    },
    "Email": {
        "icon": "✉️",
        "text": "mail@ktchn.com",
        "description": "Lorem ipsum dolor sit"
    },
    "Location": {
        "icon": "📍",
        "text": "London Eye, UK",
        "description": "Lorem ipsum dolor sit"
    }
}

with st.sidebar:
    st.page_link('streamlit_app.py', label='Introduction')
    st.page_link('pages/2_⌨️_Projects.py', label='⌨️ Projects')
    st.page_link('pages/3_📞_Contact.py', label='📞 Contact')


# Use three columns to layout the contact information
col1, col2, col3 = st.columns([1, 1, 1])

# Populate the columns with contact info
with col1:
    st.markdown(f"**{contact_info['Phone']['icon']} {contact_info['Phone']['text']}**")
    st.write(contact_info['Phone']['description'])

with col2:
    st.markdown(f"**{contact_info['Email']['icon']} {contact_info['Email']['text']}**")
    st.write(contact_info['Email']['description'])

with col3:
    st.markdown(f"**{contact_info['Location']['icon']} {contact_info['Location']['text']}**")
    st.write(contact_info['Location']['description'])
