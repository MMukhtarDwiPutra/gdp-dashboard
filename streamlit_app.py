import streamlit as st

# Fungsi untuk mengecek login
def check_login(username, password):
    if username in users and users[username] == password:
        return True
    return False

# if __name__ == '__main__':
# Data pengguna contoh (dapat disimpan di database atau file terenkripsi)
users = {"ceyy": "ceyy123"}

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

            st.write('<meta http-equiv="refresh" content="0;url=/Projects">', unsafe_allow_html=True)
        else:
            st.error("Username atau password salah")
else:
    st.write('<meta http-equiv="refresh" content="0;url=/Projects">', unsafe_allow_html=True)
