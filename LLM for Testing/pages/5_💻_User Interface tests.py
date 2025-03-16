from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from st_pages import hide_pages


st.set_page_config(
    initial_sidebar_state="collapsed"    
)

hide_pages(
    "homepage"
)


if st.button('Back'):
    switch_page("homepage")
    
st.title("User Interface Tests Generation")

st.write('Select the task')

col1,col2=st.columns(2)

with col1:
    if st.button('Write UI Test Cases', use_container_width=True):
        switch_page("UI Test Case Gen")

with col2:
    if st.button('Generate UI Test Scripts', use_container_width=True):
        switch_page("UI Test Script Gen")
