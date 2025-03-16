import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from dotenv import load_dotenv
import os


st.set_page_config(
    page_title="LLMs for Testing",
    page_icon="🖥️",
    initial_sidebar_state="collapsed"
    
)



st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("LLMs for Testing")

st.markdown('**Select the LLM Model to be used**')

model = st.selectbox('Select the LLM Model to be used',('GPT-3.5 Turbo', 'GPT-4','Google Gemini Pro'),key = 'llmModel',index=None,label_visibility="collapsed")
st.session_state['model'] = model

st.divider()

disabledButton = False

if model == None:
    disabledButton = True

st.write('Please Select the type of tests you want to generate')

col1,col2 =st.columns(2)


with col1:
    if st.button('API Tests',disabled=disabledButton,key='APIbtn', use_container_width=True):
        switch_page("API Tests")
        
with col2:
    if st.button('User Interface Tests',disabled=disabledButton, key='UIbtn', use_container_width=True):
        switch_page("User Interface tests")

if disabledButton:
    st.markdown(':red[**Please Choose the LLM to go Forward.**]')


