import streamlit as st
from dbox_aux import read_dbx_file


def run_streamlit_ui():
    st.write('Streamlit started')
    st.dataframe(read_dbx_file("/test.csv"))


