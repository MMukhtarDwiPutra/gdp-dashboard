import streamlit as st

def main():
    # builds the sidebar menu
    with st.sidebar:
        st.page_link('streamlit_app.py', label='Introduction')
        st.page_link('pages/2_⌨️_Projects.py', label='⌨️  Projects')
        st.page_link('pages/3_📞_Contact.py', label='📞  Contact')

    st.title(f'Introduction')

    #content
    st.image("data/metrotv.png", use_column_width=True)
    
    st.write("Memulai perjalanan penyiarannya dengan visi untuk menghadirkan berita yang cepat dan berkualitas. Kini, Metro TV dikenal sebagai kekuatan utama dalam media berita.")
    
    st.title("*ABOUT US*")
    st.write("Kami berfokus pada prediksi rating dan share program TV berdasarkan waktu tayang.")

if __name__ == '__main__':
    main()
