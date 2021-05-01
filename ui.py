import streamlit as st
from dbox_aux import read_dbx_file
import pandas as pd
import numpy as np
from itertools import cycle
import base64

import plotly.figure_factory as ff
import plotly.express as px
import altair as alt

def run_streamlit_ui():
    st.set_page_config(layout="wide")
    def download_link(object_to_download, download_filename, download_link_text):
        if isinstance(object_to_download,pd.DataFrame):
            object_to_download = object_to_download.to_csv(index=False)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

    main_filter = st.sidebar.selectbox('Select Option?', ('Market','Deals','Technicals','Fundamental','Valuation','Commodity','GSec'))

    nifty_indices_df = read_dbx_file("/nifty_indices.csv")
    large_deal_df = read_dbx_file("/large_deals.csv")
    technical_trigger_df = read_dbx_file("/technical_trigger.csv")
    global_pe_df = read_dbx_file("/global_pe.csv")
    comm_perf_df = read_dbx_file("/comm_perf.csv")
    global_gsec_df = read_dbx_file("/global_gsec.csv")
    test_df = read_dbx_file("/test.csv")


    if main_filter == 'Market':    
        perf_option = st.sidebar.radio('Filter Options', ['1Day Performance', 'Relative Performance'])    
        temp_dn = nifty_indices_df.iloc[:,:8].fillna(0)
        if st.button('Download'):
            tmp_download_link = download_link(temp_dn, 'Nifty Indices.csv', 'Click here to download!')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
        if perf_option == '1Day Performance':
            dt = nifty_indices_df['date'].iloc[0]
            symbols = ((np.asarray(nifty_indices_df['name'])).reshape(8,8))
            closes = ((np.asarray(nifty_indices_df['close'])).reshape(8,8))
            changes = ((np.asarray(nifty_indices_df['1D'])).reshape(8,8))
            result = pd.pivot(nifty_indices_df, index='seq1', columns='seq2', values='1D')
            labels = (np.asarray(['{0} <br> {1:,.0f} <br> {2:.2f}%'.format(s, v, c) for s, v, c in zip(symbols.flatten(), closes.flatten(), changes.flatten())])).reshape(8,8)
            
            z_flat = np.sort(changes, axis=None)
            negative_numbers = np.sum(np.array(z_flat) <= 0, axis=0)
            def normalize(x):
                return (x - np.min(x)) / (np.max(x) - np.min(x))
            normalized_scale = normalize(z_flat)
            border1 = normalized_scale[negative_numbers]
            border2 = border1 * 1.1

            mycolors=[[0, 'red'],        
                [border1, 'red'],
                [border2, 'green'], 
                [1, 'green']]
            fig1 = ff.create_annotated_heatmap(changes, annotation_text=labels, colorscale=mycolors, hoverinfo='skip')
            
            for i in range(len(fig1.layout.annotations)):
                fig1.layout.annotations[i].font.size = 11
                fig1.layout.annotations[i].font.color = 'White'
            fig1['layout']['yaxis']['autorange'] = "reversed"
            fig1.update_layout(title = f'<b>Nifty Indices Performance - {dt}</b>', width=1200, height=900, showlegend=False)
            st.plotly_chart(fig1, config= {'displaylogo': False})
        else:
            st.dataframe(nifty_indices_df.iloc[:,:8].fillna(0).set_index('name').style.background_gradient(cmap='RdYlGn', axis=0).format("{:,.1f}"), width=800, height=800)

    elif main_filter == 'Deals': 

        st.table(large_deal_df.drop(['Script Code', 'Quantity'], axis=1).set_index('Date'))
    elif main_filter == 'Technicals':    
        st.sidebar.write('Filter Options')
        trend_list = st.sidebar.multiselect('Select parameter to filter result', technical_trigger_df['trend'].drop_duplicates().sort_values().to_list(), default=technical_trigger_df['trend'].drop_duplicates().sort_values().to_list())
        trigger_list = st.sidebar.multiselect('Select parameter to filter result', technical_trigger_df['trigger'].drop_duplicates().sort_values().to_list(), default=technical_trigger_df['trigger'].drop_duplicates().sort_values().to_list())
        st.write(f"Techical trigger >>>> {' or '.join(str(x) for x in trigger_list)}")
        st.dataframe(technical_trigger_df.query("trigger == @trigger_list & trend == @trend_list")[['nse','name','trigger','industry','close','52W High','52W Low']].rename(columns={'nse':'NSE Symbol','name':'Name','trigger':'Trigger','industry':'Industry','close':'Close'}).round(0).set_index('NSE Symbol'), height=800)
        
    elif main_filter == 'Fundamental':
        st.write('Work in Progress, Please comeback later')
        st.dataframe(test_df) 
        

    elif main_filter == 'Valuation':
        global_valuation_fig = px.choropleth(global_pe_df, locations="iso_alpha",
        color="pe",
        hover_name="country",
        color_continuous_scale='reds',
        labels={'pe':'P/E Ratio'})
        global_valuation_fig.update_layout(title = '<b>Global Valuation</b>', width=1200, height=900, showlegend=False)
        st.plotly_chart(global_valuation_fig, use_container_width=True)
    elif main_filter == 'Commodity':
        st.write('Global Commodity Performance in %')
        st.dataframe(comm_perf_df.style.background_gradient(cmap='RdYlGn', axis=0).format("{:,.1f}"), width=800, height=800)
        st.write('Due to discountinuation of Baltic Dry, data is not comparabe for 1Y period')
    elif main_filter == 'GSec':
        st.write('Govt Securities 10Y Benchmarks (%)')
        global_gsec_df['Date'] = pd.to_datetime(global_gsec_df['Date'])
        
        cols = st.beta_columns(4)
        seq = cycle([0,1,2,3])
        seq1 = [next(cycle(seq)) for count in range(len(global_gsec_df['variable'].unique()))]
        for i, each in zip(seq1, global_gsec_df['variable'].unique()):
            temp = global_gsec_df.query("variable == @each")
            gsec_chart = alt.Chart(temp).mark_line().encode(x='Date', y='value').properties(title=each).configure_axis(grid=False, titleOpacity=0).configure_view(strokeOpacity=0)
            cols[i].altair_chart(gsec_chart, use_container_width=True)

    else:
        st.error("Something has gone terribly wrong.")







