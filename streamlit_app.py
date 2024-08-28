import streamlit as st

def main():
    # builds the sidebar menu
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
    # Data pengguna contoh (dapat disimpan di database atau file terenkripsi)
    users = {"ceyy": "ceyy123"}
    
    # Fungsi untuk mengecek login
    def check_login(username, password):
        if username in users and users[username] == password:
            return True
        return False

    # Jika belum login, tampilkan form login
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.title("Login")
    
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
    
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Login berhasil!")
            else:
                st.error("Username atau password salah")

    if st.session_state.logged_in:
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
    
        st.title(f'Introduction')
    
        #content
        st.image("data/metrotv.png", use_column_width=True)
        
        st.write("Memulai perjalanan penyiarannya dengan visi untuk menghadirkan berita yang cepat dan berkualitas. Kini, Metro TV dikenal sebagai kekuatan utama dalam media berita.")
        
        st.title("*ABOUT US*")
        st.write("Kami berfokus pada prediksi rating dan share program TV berdasarkan waktu tayang.")

if __name__ == '__main__':
    main()
