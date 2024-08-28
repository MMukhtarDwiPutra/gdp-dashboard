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
