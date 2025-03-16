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


st.set_page_config(
    initial_sidebar_state="collapsed",
    layout="wide"
)

hide_pages(
    "homepage"
)

 
if st.button('Back'):
    switch_page("API Test Case Gen")

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

mainBusinessObject = ""
subBusinessObject = ""
mandatoryHeaderPar = ""
nonMandatoryHeaderPar = ""
mandatoryRequestpayloadPar = ""
nonMandatoryRequestpayloadPar = ""



if 'mainBusinessObjective' in st.session_state:
    mainBusinessObject = st.session_state.mainBusinessObjective

if 'subBusinessObjective' in st.session_state:
    subBusinessObject = st.session_state.subBusinessObjective

if 'mandatoryHeaderParams' in st.session_state:
    mandatoryHeaderPar = st.session_state.mandatoryHeaderParams

if 'nonMandatoryHeaderParams' in st.session_state:
    nonMandatoryHeaderPar = st.session_state.nonMandatoryHeaderParams

if 'mandatoryRequestPayloadParameters' in st.session_state:
    mandatoryRequestpayloadPar = st.session_state.mandatoryRequestPayloadParameters

if 'nonMandatoryRequestPayloadParameters' in st.session_state:
    nonMandatoryRequestpayloadPar = st.session_state.nonMandatoryRequestPayloadParameters




st.title('Generate Java Test Scripts for API Testing')

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



if 'response' in st.session_state:
    tcResponse = st.session_state.response
    st.write('Please copy the test cases you want from the previously generated test cases below.')
    st.code(tcResponse)

st.write('Please fill the details below with respect to the test script you want to be generated')



template: str = """
I want to generate a test script for below test case. Below I mentioned the test case and API details\n
Test case type - {testCaseType}\n
Test casesd - {testCase}\n
Details:\n
API Endpoint- {apiEndpoint}\n
API Name - {apiName}\n
HTTP Method of API - {httpMethod}\n
Main Business Objective of API - {mainBusinessObjective}\n
Sub Business Objectives of API - {subBusinessObjective}\n
Mandatory Header Parameters - {mandatoryHeaderParams}\n
Non-Mandatory Header Parameters - {nonMandatoryHeaderParams}\n
Mandatory Request Payload Parameters - {mandatoryRequestPayloadParameters}\n
Non Mandatory Request Payload Parameters - {nonMandatoryRequestPayloadParameters}\n
Mandatory Response Payload Parameters - {mandatoryResponsePayloadParameters}\n
Non Mandatory Response Payload Parameters - {nonMandatoryResponsePayloadParameters}\n
Think you are a QA engineer. You need to mainly consider above mentioned test case and generate a Java script according to that testcase, as a professional QA engineer. When you write script please follow coding best practices, coding standards, exception handling as a QA engineer.  Only answer me with the code and nothing else. don't give additional text with answer. I need code only as answer.
"""



on = st.toggle('Populate fields with a sample scenario')


if not on:

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx']) 

    def get_excel_file_content_as_binary(file_path):
        with open(file_path, "rb") as file:
            return file.read()

    file_path = 'template_files/Java_Script_Code_Template.xlsx'

    excel_file_content = get_excel_file_content_as_binary(file_path)

    st.download_button(label="Download Java Script Code Template",
                    data=excel_file_content,
                    file_name="Java_Script_Code_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    uploaded_file = None

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        data = df.iloc[0]  # Assuming data for form defaults is in the first row

        with st.form('api_ts_gen'):
            st.text_input('Test Case Type', value=data.iloc[0], placeholder='Enter Test Case Type', key='testCaseType', help="Enter the type of the test case here. Ex: Positive, Negative etc.")
            st.text_area('Test Cases', value=data.iloc[1], placeholder='Please Type the Test Case', key='testCase', help="Please Enter the Test Case to be tested here.")
            st.text_input('API Endpoint', value=data.iloc[2], placeholder='Enter the API Endpoint', key='apiEndpoint', help="Please Enter the URL of the API to be tested in this field.")
            st.text_input('API Name', value=data.iloc[3], placeholder='Enter API Name', key='apiName', help="Please Enter the Name of the API Endpoint here.")
            st.text_input('HTTP Method of API', value=data.iloc[4], placeholder='Enter the HTTP Method', key='httpMethod', help="Please Enter the HTTP method of the API Ex: POST, GET, DELETE, PUT etc.")
            st.text_input('Type of End Users', value=data.iloc[5], placeholder='Enter the type of End Users', key='endUserType', help="Enter the type of the End Users as per their roles.")
            st.text_input('Main Business Objective of API', value=data.iloc[6], placeholder='', key='mainBusinessObjective', help="Enter the Primary Business Objective to be tested.")
            st.text_area('Sub Business Objectives of API', value=data.iloc[7], placeholder='', key='subBusinessObjective', help="Enter Sub Business Objectives to be tested. These objectives should be secondary objectives than the Primary Objective.")
            st.text_input('Mandatory Header Parameters', value=data.iloc[8], placeholder='', key='mandatoryHeaderParams')
            st.text_input('Non-Mandatory Header Parameters', value=data.iloc[9], placeholder='', key='nonMandatoryHeaderParams')
            st.text_input('Mandatory Request Payload Parameters', value=data.iloc[10], placeholder='', key='mandatoryRequestPayloadParameters')
            st.text_input('Non-Mandatory Request Payload Parameters', value=data.iloc[11], placeholder='', key='nonMandatoryRequestPayloadParameters')
            st.text_input('Mandatory Response Payload Parameters', value=data.iloc[12], placeholder='', key='mandatoryResponsePayloadParameters')
            st.text_input('Non-Mandatory Response Payload Parameters', value=data.iloc[13], placeholder='', key='nonMandatoryResponsePayloadParameters')

            submitted = st.form_submit_button("Generate")
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")


elif on:
    with st.form('api_ts_gen'):
        st.text_input('Test Case Type', placeholder='Enter Test Case Type', value= 'Positive Test Case',key = 'testCaseType',help="Enter the type of the test case here. Ex: Positive, Negative etc.")
        st.text_area('Test Cases', placeholder='Please Type the Test Case', value= 'Verify that a meeting can be scheduled successfully with a valid RM and customer.',key = 'testCase', help="Please Enter the Test Case to be tested here.")
        st.text_input('API Endpoint', placeholder='Enter the API Endpoint', value= 'https://localhost/api/v1/test',key = 'apiEndpoint', help="Please Enter the URL of the API to be tested in this field.")
        st.text_input('API Name', placeholder='Enter API Name', key = 'apiName', value='ABC Application Meeting Scheduling Endpoint with Bankers and Gold customers',help="Please Enter the Name of the API Endpoint here.")
        st.text_input('HTTP Method of API', placeholder='Enter the HTTP Method', value='POST',key = 'httpMethod', help="Please Enter the HTTP method of the API Ex: POST, GET, DELETE, PUT etc.")
        st.text_input('Type of End Users', placeholder='Enter the type of End Users', value='Two types of Bankers named Relationship Managers and Financial Advisors',key = 'endUserType', help="Enter the type of the End Users as per their roles." )
        st.text_input('Main Business Objective of API', placeholder='',value=' Schedule a meeting with a Banker and  a customer in a given time in the ABC application', key = 'mainBusinessObjective', help="Enter the Primary Business Objective to be tested." )
        st.text_area('Sub Business Objectives of API', placeholder='', value="""An account for RM should be created in the ABC application if not available already, An account for FA should be created in the ABC application if not available already, If an account is not present for the customer in the ABC application it should be created automatically, If relevant RM and Customer are not mapped in the ABC Application a mapping should be created between RM and customer when a meeting is scheduled between the two parties for the first time, A mapping should be created between the RM and Customer if relevant FA and Customer are not mapped in the application, A Customer should be Mapped with FA and RM when a meeting is scheduled with both RM and FA. """,key = 'subBusinessObjective', help="Enter Sub Business Objectiives to be tested. These objectives should be secondary objectives than the Primary Objective.")
        # st.text_input('Test Scenario Combination', placeholder='', key = 'testScenarioCombination')
        st.text_input('Mandatory Header Parameters', placeholder='',value= 'Country code and Business Code ', key = 'mandatoryHeaderParams')
        st.text_input('Non-Mandatory Header Parameters', placeholder='', value= 'UUID',key = 'nonMandatoryHeaderParams')
        st.text_input('Mandatory Request Payload Parameters', placeholder='',value= 'Banker ID, Customer ID, Banker First Name, Customer First Name, Host Type, Start Time and End Time ', key = 'mandatoryRequestPayloadParameters')
        st.text_input('Non-Mandatory Request Payload Parameters', placeholder='',value= 'Banker Last Name and Customer Last name ', key = 'nonMandatoryRequestPayloadParameters')
        st.text_input('Mandatory Response Payload Parameters', placeholder='',value= 'N/A',  key = 'mandatoryResponsePayloadParameters')
        st.text_input('Non-Mandatory Response Payload Parameters', placeholder='', value= 'N/A',  key = 'nonMandatoryResponsePayloadParameters')
        #st.selectbox('Select the Test Script Language',('Python', 'Java', 'Cucumber'),key = 'language',index=0)
        #st.selectbox('Rest Assured Type',('Rest Assured','Rest Assured with BDD'),key= 'restAssuredType',index=0)
        submitted = st.form_submit_button("Generate")
else: 
    with st.form('api_ts_gen'):
        st.text_input('Test Case Type', placeholder='Enter Test Case Type', key = 'testCaseType',help="Enter the type of the test case here. Ex: Positive, Negative etc.")
        st.text_area('Test Cases', placeholder='Please Type the Test Case', key = 'testCase', help="Please Enter the Test Case to be tested here.")
        st.text_input('API Endpoint', placeholder='Enter the API Endpoint', key = 'apiEndpoint', help="Please Enter the URL of the API to be tested in this field.")
        st.text_input('API Name', placeholder='Enter API Name', key = 'apiName', help="Please Enter the Name of the API Endpoint here.")
        st.text_input('HTTP Method of API', placeholder='Enter the HTTP Method', key = 'httpMethod', help="Please Enter the HTTP method of the API Ex: POST, GET, DELETE, PUT etc.")
        st.text_input('Type of End Users', placeholder='Enter the type of End Users', key = 'endUserType', help="Enter the type of the End Users as per their roles." )
        st.text_input('Main Business Objective of API',  placeholder='', value= mainBusinessObject,  key = 'mainBusinessObjective', help="Enter the Primary Business Objective to be tested." )
        st.text_area('Sub Business Objectives of API', placeholder='', value= subBusinessObject, key = 'subBusinessObjective', help="Enter Sub Business Objectiives to be tested. These objectives should be secondary objectives than the Primary Objective.")
        # st.text_input('Test Scenario Combination', placeholder='', key = 'testScenarioCombination')
        st.text_input('Mandatory Header Parameters', placeholder='',value= mandatoryHeaderPar, key = 'mandatoryHeaderParams')
        st.text_input('Non-Mandatory Header Parameters', placeholder='', value= nonMandatoryHeaderPar,key = 'nonMandatoryHeaderParams')
        st.text_input('Mandatory Request Payload Parameters', placeholder='',value= mandatoryRequestpayloadPar, key = 'mandatoryRequestPayloadParameters')
        st.text_input('Non-Mandatory Request Payload Parameters', placeholder='',value= nonMandatoryRequestpayloadPar, key = 'nonMandatoryRequestPayloadParameters')
        st.text_input('Mandatory Response Payload Parameters', placeholder='', key = 'mandatoryResponsePayloadParameters')
        st.text_input('Non-Mandatory Response Payload Parameters', placeholder='', key = 'nonMandatoryResponsePayloadParameters')
        #st.selectbox('Select the Test Script Language',('Python', 'Java', 'Cucumber'),key = 'language',index=0)
        #st.selectbox('Rest Assured Type',('Rest Assured','Rest Assured with BDD'),key= 'restAssuredType',index=0)
        submitted = st.form_submit_button("Generate")




if submitted: 
    api_ts_template = PromptTemplate.from_template(template)
    api_ts_template.input_variables=['testCaseType','testCase','apiEndpoint','apiName','httpMethod','endUserType','mainBusinessObjective','subBusinessObjective','mandatoryHeaderParams','nonMandatoryHeaderParams','mandatoryRequestPayloadParameters','nonMandatoryRequestPayloadParameters','mandatoryResponsePayloadParameters','nonMandatoryResponsePayloadParameters']

    formatted_prompt = api_ts_template.format(
        testCaseType = st.session_state.testCaseType,
        testCase = st.session_state.testCase,
        apiEndpoint = st.session_state.apiEndpoint,
        apiName = st.session_state.apiName,
        httpMethod = st.session_state.httpMethod,
        mainBusinessObjective = st.session_state.mainBusinessObjective,
        subBusinessObjective = st.session_state.subBusinessObjective,
        # testScenarioCombination = st.session_state.testScenarioCombination,
        mandatoryHeaderParams = st.session_state.mandatoryHeaderParams,
        nonMandatoryHeaderParams = st.session_state.nonMandatoryHeaderParams,
        mandatoryRequestPayloadParameters = st.session_state.mandatoryRequestPayloadParameters,
        nonMandatoryRequestPayloadParameters = st.session_state.nonMandatoryRequestPayloadParameters,
        mandatoryResponsePayloadParameters = st.session_state.mandatoryResponsePayloadParameters,
        nonMandatoryResponsePayloadParameters = st.session_state.nonMandatoryResponsePayloadParameters
    )

            

    if model == 'GPT-3.5 Turbo':

        st.write('Using: '+model)

        llm = OpenAI(model_name= "gpt-3.5-turbo-0613", temperature = 0.5)

        if(len(formatted_prompt) != 0):
            response = llm(formatted_prompt)
            st.session_state['response_code'] = response
            # st.code(response)
            # if 'response' in st.session_state:
            #     del st.session_state['response']

    if model ==  'GPT-4': 
        st.write('Using: ' + model)

        llm = ChatOpenAI(model_name= "gpt-4o", temperature = 0, model_kwargs={"seed": 10})

        if(len(formatted_prompt) != 0):
            response = llm.invoke(formatted_prompt)
            st.session_state['response_code'] = response.content
            # st.code(response.content)
            # if 'response' in st.session_state:
            #     del st.session_state['response']
            
        

    if model ==  'Google Gemini Pro': 
        st.write('Using: ' + model)

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)

        if(len(formatted_prompt) != 0):
            response = llm.invoke(formatted_prompt)
            st.session_state['response_code'] = response.content
            # st.code(response.content)
            # if 'response' in st.session_state:
            #     del st.session_state['response']
            

    if model == None:
        st.error('Please Select a LLM')

def describe_code_function(response, llm):

    full_prompt = f"""
        Generated Code : {response}

        Think you as the Expert Automation Quality Assurance Engineer, Understand above given java api test script code. 
        Think step by step and describe the code in detail.
        Give the code with adding description for each line of the code as comments.
    """
    if model == 'GPT-3.5 Turbo':
        llm = OpenAI(model_name= "gpt-3.5-turbo-0613", temperature = 0.1)
        response = llm(full_prompt)
        return response
    
    if model ==  'GPT-4':
        llm = ChatOpenAI(model_name= "gpt-4o", temperature = 0, model_kwargs={"seed": 10})
        response = llm.invoke(full_prompt)
        return response.content
    
    if model ==  'Google Gemini Pro':
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)
        response = llm.invoke(full_prompt)
        return response.content
    

if 'response_code' in st.session_state:
    st.code(st.session_state['response_code'])

    
    file_extension = 'java'

    filename = f"api_test_script_code.{file_extension}"
    
    code_to_download = str(st.session_state['response_code']) 
    
    col1, col2 = st.columns(2)

    with col1: 
        st.download_button(
            label="Export Code",
            data=code_to_download,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

    with col2: 
        describe_code = st.button(
            label="Describe the Code",
            use_container_width=True,
            key="describe_code"
        )


    if describe_code:
        full_description = describe_code_function(st.session_state['response_code'], model)
        st.code(full_description)

st.cache_data.clear()