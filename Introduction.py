import streamlit as st

st.set_page_config(
    page_title="Multipage App",
)

st.sidebar.title("Introduction")
st.sidebar.success("Select a page above")


st.image("data/metrotv.png", use_column_width=True)

st.write("Memulai perjalanan penyiarannya dengan visi untuk menghadirkan berita yang cepat dan berkualitas. Kini, Metro TV dikenal sebagai kekuatan utama dalam media berita.")

st.title("*ABOUT US*")
st.write("Kami berfokus pada prediksi rating dan share program TV berdasarkan waktu tayang.")
