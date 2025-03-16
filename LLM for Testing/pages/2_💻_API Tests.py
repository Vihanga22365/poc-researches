from streamlit_extras.switch_page_button import switch_page
import streamlit as st
from st_pages import hide_pages


st.set_page_config(
    initial_sidebar_state="collapsed"

)



hide_pages("homepage")







if st.button('Back'):
    switch_page("homepage")

st.title("API Tests Generation")



st.write('Select the task')

col1,col2, col3=st.columns(3)

        
with col1:
    if st.button('Write API Test Cases', use_container_width=True):
        switch_page("API Test Case Gen")
        
with col2:
    if st.button('Generate Unit Tests', use_container_width=True):
        switch_page("Unit Tests")

with col3:
    if st.button('Generate Test Scripts', use_container_width=True):
        switch_page("API Test Script Gen")
