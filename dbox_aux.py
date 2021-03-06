import pandas as pd
import io
import dropbox
import streamlit as st

TOKEN = st.secrets["TOKEN"]
dbx = dropbox.Dropbox(TOKEN)

def read_dbx_file(file):
    print('Getting latest file')
    _, f = dbx.files_download(file)
    with io.BytesIO(f.content) as stream:
        df = pd.read_csv(stream, index_col=0)
    return df
