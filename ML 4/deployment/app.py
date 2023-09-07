import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title='Customer Analysis',
    layout='centered',
    initial_sidebar_state= 'auto'
    )

navbar = st.sidebar.selectbox('#### Page:',('Analysis','Cluster'))

if navbar =='Analysis':
    eda.run()
elif navbar == 'Cluster':
    prediction.run()