import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import os
from st_pages import hide_pages
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
st.set_page_config(
    initial_sidebar_state="collapsed",
    layout="wide"
)

hide_pages(
    "homepage"
)

 
if st.button('Back'):
    switch_page("User Interface tests")

st.title('Generate Test Cases for Functional UI Testing')

if 'model' in st.session_state:
    model = st.session_state.model
    st.write('Selected LLM: ',model)
    if st.button('Change the LLM',help='Change the LLM'):
        #switch_page('homepage')
        del st.session_state['model']
        model = st.selectbox(':red[Select the LLM Model to be used]',('GPT-3.5 Turbo', 'GPT-4','Google Gemini Pro'),key = 'llmModel', index = None)

        if model != None:
            st.session_state.model = model

else:
    model = st.selectbox(':red[Select the LLM Model to be used]',('GPT-3.5 Turbo', 'GPT-4','Google Gemini Pro'),key = 'llmModel', index = None)

    if model != None:
        st.session_state.model = model

st.write('Please fill the details below with respect to the test case you want to be generated')




template = """ 
I want to generate Positive and negative test cases to verify the following define business requirement\n
User Story Name - {userStoryName}\n
Main Business Functionality - {mainBusinessFunc}\n
Sub Business Functionalites - {subBusinessFunc}\n
Precondition - {precondition}\n
Type of End Users - {endUsersType}\n
Think you are a QA engineer. Generate all possible positive and negative test cases listed separately as a professional QA engineer to the given objective details. When writing the test cases, follow QA standards, and keywords. Please give test cases as a single line.
"""


on = st.toggle('Populate fields with a sample scenario')


if not on:

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx']) 

    def get_excel_file_content_as_binary(file_path):
        with open(file_path, "rb") as file:
            return file.read()

    file_path = 'template_files/UI_Test_Cases_Template.xlsx'

    excel_file_content = get_excel_file_content_as_binary(file_path)

    st.download_button(label="Download UI Test Case Template",
                    data=excel_file_content,
                    file_name="UI_Test_Cases_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    uploaded_file = None

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        data = df.iloc[0]  # Assuming data for form defaults is in the first row

        with st.form('user_story_form'):
            st.text_input('User Story Name', value=data.iloc[0], key='userStoryName', help="Please Enter the Name of the User Story here")
            st.text_area('Main Business Functionality', value=data.iloc[1], key='mainBusinessFunc', help="Enter the Primary Business Functionality to be tested.")
            st.text_area('Sub Business Functionalities', value=data.iloc[2], key='subBusinessFunc', help="Enter the Sub business functionalities to be tested.")
            st.text_input('Precondition', value=data.iloc[3], key='precondition', help="Please Enter the Pre Conditions that should be met.")
            st.text_input('Type of End Users', value=data.iloc[4], key='endUsersType', help="Enter the type of the End Users as per their roles.")

            submitted = st.form_submit_button('Generate')
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")


elif on:  
    with st.form('api_tc_gen', clear_on_submit=True):
        st.text_input('User Story Name', placeholder='Enter the User Story Name',value='Agent and Gold Customer Meeting Scheduling in the ABC Application', key = 'userStoryName',help="Please Enter the Name of the User Story here")
        st.text_area('Main Business Functionality', placeholder='', value='The meeting details are added to the relevant agent timeline and calendar, The meeting details are added to the relevant customer calendar and dashboard, The agent can initiate the meeting in ABC Application and the customer should be able to join the meeting.',key = 'mainBusinessFunc',help="Enter the Primary Business Functionality to be tested.")
        st.text_area('Sub Business Functionalities', placeholder='', value='N/A',key = 'subBusinessFunc',help="Enter the Sub business functionalities to be tested.")
        #st.text_input('Test Scenario Combination', placeholder='', key = 'testScenarioCombination')
        st.text_input('Precondition', placeholder='Precondition',value='The meeting request should come from XYZ end.', key = 'precondition', help="Please Enter the Pre Conditions that should be met.")
        st.text_input('Type of End Users', placeholder='Type of End Users', value='Two types of agents named Relationship Managers and Financial Advisors' ,key = 'endUsersType', help="Enter the type of the End Users as per their roles.")
        submitted = st.form_submit_button('Generate')

else: 
    with st.form('api_tc_gen', clear_on_submit=True):
        st.text_input('User Story Name', placeholder='Enter the User Story Name', key = 'userStoryName',help="Please Enter the Name of the User Story here")
        st.text_area('Main Business Functionality', placeholder='', key = 'mainBusinessFunc',help="Enter the Primary Business Functionality to be tested.")
        st.text_area('Sub Business Functionalities', placeholder='', key = 'subBusinessFunc',help="Enter the Sub business functionalities to be tested.")
        #st.text_input('Test Scenario Combination', placeholder='', key = 'testScenarioCombination')
        st.text_input('Precondition', placeholder='Precondition', key = 'precondition', help="Please Enter the Pre Conditions that should be met.")
        st.text_input('Type of End Users', placeholder='Type of End Users', key = 'endUsersType', help="Enter the type of the End Users as per their roles.")
        submitted = st.form_submit_button('Generate')

if submitted: 
    ui_tc_template = PromptTemplate.from_template(template)
    ui_tc_template.input_variables = ['userStoryName','mainBusinessFunc','subBusinessFunc','precondition','endUsersType']


    formatted_prompt = ui_tc_template.format(
        userStoryName = st.session_state.userStoryName,
        mainBusinessFunc = st.session_state.mainBusinessFunc,
        subBusinessFunc = st.session_state.subBusinessFunc,
        # testScenarioCombination = st.session_state.testScenarioCombination,
        precondition = st.session_state.precondition,
        endUsersType = st.session_state.endUsersType
    )



    if model == 'GPT-3.5 Turbo':

        st.write('Using: '+model)

        llm = OpenAI(model_name= "gpt-3.5-turbo-0613", temperature = 0.5)

        if(len(formatted_prompt) != 0):
            response = llm(formatted_prompt)
            st.code(response)
            
    if model ==  'GPT-4': 
        st.write('Using: ' + model)

        llm = ChatOpenAI(model_name= "gpt-4o", temperature = 0, model_kwargs={"seed": 10})

        if(len(formatted_prompt) != 0):
            response = llm.invoke(formatted_prompt)
            st.code(response.content)
 

    if model ==  'Google Gemini Pro': 
        st.write('Using: ' + model)

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)

        if(len(formatted_prompt) != 0):
            response = llm.invoke(formatted_prompt)
            st.code(response.content)
            

    if model == None:
        st.error('Please Select a LLM')







