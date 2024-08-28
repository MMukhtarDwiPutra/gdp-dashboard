import streamlit as st
from PIL import Image

if st.button("Logout"):
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.write('<meta http-equiv="refresh" content="0;url=/">', unsafe_allow_html=True)
else:
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
