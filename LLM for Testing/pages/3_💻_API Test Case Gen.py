import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from st_pages import hide_pages
import pandas as pd


st.set_page_config(
    initial_sidebar_state="collapsed",
    layout="wide"
)

hide_pages(
    "homepage"
)

if st.button('Back'):
    switch_page("API Tests")

hide_pages(
    "homepage"
)


load_dotenv()

resultStatus = True

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.title('Generate Test Cases for API Testing')



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



template: str = """
I want to generate Positive and negative test cases to verify the following API with define business requirement and expected result of each test cases\n
API Name - {apiName}\n
HTTP Method of API - {httpMethod}\n
Main Business Objective of API - {mainBusinessObjective}\n
Sub Business Objectives of API - {subBusinessObjective}\n
Mandatory Request Payload Parameters - {mandatoryHeaderParams}\n
Respective mandatory parameter Value - {respectiveMandatoryParam}\n
Non-Mandatory Header Parameters - {nonMandatoryHeaderParams}\n
Respective non-mandatory parameter Value - {respectiveNonMandatoryParam}\n
Mandatory Request Payload Parameters - {mandatoryRequestPayloadParameters}\n
Non Mandatory Request Payload Parameters - {nonMandatoryRequestPayloadParameters}\n
Mandatory Response Payload Parameters - {mandatoryResponsePayloadParameters}\n
Non Mandatory Response Payload Parameters - {nonMandatoryResponsePayloadParameters}\n

Think you are a QA engineer. Generate all possible positive and negative test cases list separately as a professional QA engineer to the given business objective. 
When writing the test cases, follow  QA standards, and keywords. Please write test cases as single line.
"""



on = st.toggle('Populate fields with a sample scenario')



if not on:

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx']) 

    def get_excel_file_content_as_binary(file_path):
        with open(file_path, "rb") as file:
            return file.read()

    file_path = 'template_files/Backend_Test_Cases_Template.xlsx'

    excel_file_content = get_excel_file_content_as_binary(file_path)

    st.download_button(label="Download Backend Test Case Template",
                    data=excel_file_content,
                    file_name="Backend_Test_Cases_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    uploaded_file = None

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        data = df.iloc[0]

        with st.form('api_tc_gen'):
            st.text_input('API Name', value=data.iloc[0], key='apiName', help="Please Enter the Name of the API Endpoint here.")
            st.text_input('HTTP Method of API', value=data.iloc[1], key='httpMethod', help="Please Enter the HTTP method of the API Ex: POST, GET, DELETE, PUT etc.")
            st.text_input('Type of End Users', value=data.iloc[2], key='endUserType', help="Enter the type of the End Users as per their roles.")
            st.text_input('Main Business Objective of API', value=data.iloc[3], key='mainBusinessObjective', help="Enter the Primary Business Objective to be tested.")
            st.text_area('Sub Business Objectives of API', value=data.iloc[4], key='subBusinessObjective', help="Enter Sub Business Objectives to be tested. These objectives should be secondary objectives than the Primary Objective.")
            st.text_input('Mandatory Header Parameters', value=data.iloc[5], key='mandatoryHeaderParams')
            st.text_input('Respective Mandatory Parameter Value', value=data.iloc[6], key='respectiveMandatoryParam')
            st.text_input('Non-Mandatory Header Parameters', value=data.iloc[7], key='nonMandatoryHeaderParams')
            st.text_input('Respective Non-Mandatory Parameter Value', value=data.iloc[8], key='respectiveNonMandatoryParam')
            st.text_input('Mandatory Request Payload Parameters', value=data.iloc[9], key='mandatoryRequestPayloadParameters')
            st.text_input('Non-Mandatory Request Payload Parameters', value=data.iloc[10], key='nonMandatoryRequestPayloadParameters')
            st.text_input('Mandatory Response Payload Parameters', value=data.iloc[11], key='mandatoryResponsePayloadParameters')
            st.text_input('Non-Mandatory Response Payload Parameters', value=data.iloc[12], key='nonMandatoryResponsePayloadParameters')
            submitted = st.form_submit_button("Generate")

            if submitted:
                api_tc_template = PromptTemplate.from_template(template)
                api_tc_template.input_variables=['apiName','httpMethod','endUserType','mainBusinessObjective','subBusinessObjective','testScenarioCombination','mandatoryHeaderParams','respectiveMandatoryParam','nonMandatoryHeaderParams','respectiveNonMandatoryParam','mandatoryRequestPayloadParameters','nonMandatoryRequestPayloadParameters','mandatoryResponsePayloadParameters','nonMandatoryResponsePayloadParameters']

            
                
                fprompt = api_tc_template.format(
                    apiName = st.session_state.apiName,
                    httpMethod = st.session_state.httpMethod,
                    mainBusinessObjective = st.session_state.mainBusinessObjective,
                    subBusinessObjective = st.session_state.subBusinessObjective,
                    mandatoryHeaderParams = st.session_state.mandatoryHeaderParams,
                    respectiveMandatoryParam = st.session_state.respectiveMandatoryParam,
                    nonMandatoryHeaderParams = st.session_state.nonMandatoryHeaderParams,
                    respectiveNonMandatoryParam = st.session_state.respectiveNonMandatoryParam,
                    mandatoryRequestPayloadParameters = st.session_state.mandatoryRequestPayloadParameters,
                    nonMandatoryRequestPayloadParameters = st.session_state.nonMandatoryRequestPayloadParameters,
                    mandatoryResponsePayloadParameters = st.session_state.mandatoryResponsePayloadParameters,
                    nonMandatoryResponsePayloadParameters = st.session_state.nonMandatoryResponsePayloadParameters
            )



                if model == 'GPT-3.5 Turbo':

                    st.write('Using: '+model)

                    llm = OpenAI(model_name= "gpt-3.5-turbo-0613", temperature = 0.5)

                    if(len(fprompt) != 0):
                        response = llm(fprompt)
                        st.code(response)
                        st.session_state['response'] = response
                    
                    if(len(response) != 0):
                        resultStatus = False

                if model ==  'Google Gemini Pro': 
                    st.write('Using: ' + model)

                    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)

                    if(len(fprompt) != 0):
                        response = llm.invoke(fprompt)
                        st.code(response.content)
                        st.session_state['response'] = response.content

                if model ==  'GPT-4': 
                    st.write('Using: ' + model)

                    llm = ChatOpenAI(model_name= "gpt-4o", temperature = 0.5)

                    if(len(fprompt) != 0):
                        response = llm.invoke(fprompt)
                        st.code(response.content)
                        st.session_state['response'] = response.content

                    if(len(response.content) != 0):
                        resultStatus = False

                if model == None:
                    st.error('Please Select a LLM')

            st.divider()
            st.subheader('Write Test Scripts')
            st.caption(':red[Before selecting the language, please copy the required test cases to the clipboard or note them down to write the script out of the generated test cases before clicking the button.]')

            st.selectbox('Select the Test Script Language',('Python', 'Java', 'Cucumber (Feature File Already Available)','Cucumber (Generate Both Feature File and Script)'),key = 'script_lang',index=0,disabled=resultStatus)
            scripted = st.form_submit_button("Write Test Scripts", disabled=resultStatus)

            if scripted:
                if st.session_state.script_lang == 'Cucumber (Feature File Already Available)' :
                    switch_page("BDD With Feature File")
                
                if st.session_state.script_lang == 'Cucumber (Generate Both Feature File and Script)' :
                    switch_page("BDD Without Feature File")

                if st.session_state.script_lang == 'Java' :
                    switch_page("API Test Script Gen Java")

                if st.session_state.script_lang == 'Python' :
                    switch_page("API Test Script Gen Python")
                
                else :
                    switch_page("API Test Script Gen")
    except Exception as e:
        print("Error reading Excel file:", e)

elif on:
    
    with st.form('api_tc_gen'):
        st.text_input('API Name', placeholder='Enter API Name', key = 'apiName', value='ABC Application Meeting Scheduling Endpoint with Bankers and Gold customers',help="Please Enter the Name of the API Endpoint here.")
        st.text_input('HTTP Method of API', placeholder='Enter the HTTP Method', value='POST',key = 'httpMethod', help="Please Enter the HTTP method of the API Ex: POST, GET, DELETE, PUT etc.")
        st.text_input('Type of End Users', placeholder='Enter the type of End Users', value='Two types of Bankers named Relationship Managers and Financial Advisors',key = 'endUserType', help="Enter the type of the End Users as per their roles." )
        st.text_input('Main Business Objective of API', placeholder='',value=' Schedule a meeting with a Banker and  a customer in a given time in the ABC application', key = 'mainBusinessObjective', help="Enter the Primary Business Objective to be tested." )
        st.text_area('Sub Business Objectives of API', placeholder='', value="""An account for RM should be created in the ABC application if not available already, An account for FA should be created in the ABC application if not available already, If an account is not present for the customer in the ABC application it should be created automatically, If relevant RM and Customer are not mapped in the ABC Application a mapping should be created between RM and customer when a meeting is scheduled between the two parties for the first time, A mapping should be created between the RM and Customer if relevant FA and Customer are not mapped in the application, A Customer should be Mapped with FA and RM when a meeting is scheduled with both RM and FA. """,key = 'subBusinessObjective', help="Enter Sub Business Objectiives to be tested. These objectives should be secondary objectives than the Primary Objective.")
        st.text_input('Mandatory Header Parameters', placeholder=' ',value='Country code and Business Code', key = 'mandatoryHeaderParams')
        st.text_input('Respective Mandatory Parameter Value', placeholder='',value='US and GCB', key = 'respectiveMandatoryParam')
        st.text_input('Non-Mandatory Header Parameters', placeholder='',value='UUID', key = 'nonMandatoryHeaderParams')
        st.text_input('Respective Non-Mandatory Parameter Value', placeholder='', value='123456',key = 'respectiveNonMandatoryParam')
        st.text_input('Mandatory Request Payload Parameters', placeholder='',value='Banker ID, Customer ID, Banker First Name, Customer First Name, Host Type,Start Time and End Time', key = 'mandatoryRequestPayloadParameters')
        st.text_input('Non-Mandatory Request Payload Parameters', placeholder='', value='Banker Last Name and Customer Last name ',key = 'nonMandatoryRequestPayloadParameters')
        st.text_input('Mandatory Response Payload Parameters', placeholder='',value='N/A', key = 'mandatoryResponsePayloadParameters')
        st.text_input('Non-Mandatory Response Payload Parameters', placeholder='', value='N/A',key = 'nonMandatoryResponsePayloadParameters')
        submitted = st.form_submit_button("Generate")
        

        if submitted:
            api_tc_template = PromptTemplate.from_template(template)
            api_tc_template.input_variables=['apiName','httpMethod','endUserType','mainBusinessObjective','subBusinessObjective','testScenarioCombination','mandatoryHeaderParams','respectiveMandatoryParam','nonMandatoryHeaderParams','respectiveNonMandatoryParam','mandatoryRequestPayloadParameters','nonMandatoryRequestPayloadParameters','mandatoryResponsePayloadParameters','nonMandatoryResponsePayloadParameters']

        
            
            fprompt = api_tc_template.format(
                apiName = st.session_state.apiName,
                httpMethod = st.session_state.httpMethod,
                mainBusinessObjective = st.session_state.mainBusinessObjective,
                subBusinessObjective = st.session_state.subBusinessObjective,
                mandatoryHeaderParams = st.session_state.mandatoryHeaderParams,
                respectiveMandatoryParam = st.session_state.respectiveMandatoryParam,
                nonMandatoryHeaderParams = st.session_state.nonMandatoryHeaderParams,
                respectiveNonMandatoryParam = st.session_state.respectiveNonMandatoryParam,
                mandatoryRequestPayloadParameters = st.session_state.mandatoryRequestPayloadParameters,
                nonMandatoryRequestPayloadParameters = st.session_state.nonMandatoryRequestPayloadParameters,
                mandatoryResponsePayloadParameters = st.session_state.mandatoryResponsePayloadParameters,
                nonMandatoryResponsePayloadParameters = st.session_state.nonMandatoryResponsePayloadParameters
        )


            if model == 'GPT-3.5 Turbo':

                st.write('Using: '+model)

                llm = OpenAI(model_name= "gpt-3.5-turbo-0613", temperature = 0.5)

                if(len(fprompt) != 0):
                    response = llm(fprompt)
                    st.code(response)
                    st.session_state['response'] = response
                
                if(len(response) != 0):
                    resultStatus = False

            if model ==  'Google Gemini Pro': 
                st.write('Using: ' + model)

                llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)

                if(len(fprompt) != 0):
                    response = llm.invoke(fprompt)
                    st.code(response.content)
                    st.session_state['response'] = response.content

                if(len(response.content) != 0):
                    resultStatus = False

            if model ==  'GPT-4': 
                st.write('Using: ' + model)

                llm = ChatOpenAI(model_name= "gpt-4o", temperature = 0.5)

                if(len(fprompt) != 0):
                    response = llm.invoke(fprompt)
                    st.code(response.content)
                    st.session_state['response'] = response.content

                if(len(response.content) != 0):
                    resultStatus = False

            

            if model == None:
                st.error('Please Select a LLM')


        st.divider()
        st.subheader('Write Test Scripts')
        st.caption(':red[Before selecting the language, please copy the required test cases to the clipboard or note them down to write the script out of the generated test cases before clicking the button.]')

        st.selectbox('Select the Test Script Language',('Python', 'Java', 'Cucumber (Feature File Already Available)','Cucumber (Generate Both Feature File and Script)'),key = 'script_lang',index=0,disabled=resultStatus)
        scripted = st.form_submit_button("Write Test Scripts", disabled=resultStatus)

        if scripted:
            if st.session_state.script_lang == 'Cucumber (Feature File Already Available)' :
                switch_page("BDD With Feature File")
            
            if st.session_state.script_lang == 'Cucumber (Generate Both Feature File and Script)' :
                switch_page("BDD Without Feature File")

            if st.session_state.script_lang == 'Java' :
                switch_page("API Test Script Gen Java")

            if st.session_state.script_lang == 'Python' :
                switch_page("API Test Script Gen Python")
            
            else :
                switch_page("API Test Script Gen")

else:           

    with st.form('api_tc_gen'):
        st.text_input('API Name', placeholder='Enter API Name', key = 'apiName', help="Please Enter the Name of the API Endpoint here.")
        st.text_input('HTTP Method of API', placeholder='Enter the HTTP Method', key = 'httpMethod', help="Please Enter the HTTP method of the API Ex: POST, GET, DELETE, PUT etc.")
        st.text_input('Type of End Users', placeholder='Enter the type of End Users', key = 'endUserType', help="Enter the type of the End Users as per their roles." )
        st.text_input('Main Business Objective of API', placeholder='', key = 'mainBusinessObjective', help="Enter the Primary Business Objective to be tested." )
        st.text_area('Sub Business Objectives of API', placeholder='', key = 'subBusinessObjective', help="Enter Sub Business Objectiives to be tested. These objectives should be secondary objectives than the Primary Objective.")
        st.text_input('Mandatory Header Parameters', placeholder='', key = 'mandatoryHeaderParams')
        st.text_input('Respective Mandatory Parameter Value', placeholder='', key = 'respectiveMandatoryParam')
        st.text_input('Non-Mandatory Header Parameters', placeholder='', key = 'nonMandatoryHeaderParams')
        st.text_input('Respective Non-Mandatory Parameter Value', placeholder='', key = 'respectiveNonMandatoryParam')
        st.text_input('Mandatory Request Payload Parameters', placeholder='', key = 'mandatoryRequestPayloadParameters')
        st.text_input('Non-Mandatory Request Payload Parameters', placeholder='', key = 'nonMandatoryRequestPayloadParameters')
        st.text_input('Mandatory Response Payload Parameters', placeholder='', key = 'mandatoryResponsePayloadParameters')
        st.text_input('Non-Mandatory Response Payload Parameters', placeholder='', key = 'nonMandatoryResponsePayloadParameters')
        submitted = st.form_submit_button("Generate")
        

        if submitted:
            api_tc_template = PromptTemplate.from_template(template)
            api_tc_template.input_variables=['apiName','httpMethod','endUserType','mainBusinessObjective','subBusinessObjective','testScenarioCombination','mandatoryHeaderParams','respectiveMandatoryParam','nonMandatoryHeaderParams','respectiveNonMandatoryParam','mandatoryRequestPayloadParameters','nonMandatoryRequestPayloadParameters','mandatoryResponsePayloadParameters','nonMandatoryResponsePayloadParameters']

        
            
            fprompt = api_tc_template.format(
                apiName = st.session_state.apiName,
                httpMethod = st.session_state.httpMethod,
                mainBusinessObjective = st.session_state.mainBusinessObjective,
                subBusinessObjective = st.session_state.subBusinessObjective,
                mandatoryHeaderParams = st.session_state.mandatoryHeaderParams,
                respectiveMandatoryParam = st.session_state.respectiveMandatoryParam,
                nonMandatoryHeaderParams = st.session_state.nonMandatoryHeaderParams,
                respectiveNonMandatoryParam = st.session_state.respectiveNonMandatoryParam,
                mandatoryRequestPayloadParameters = st.session_state.mandatoryRequestPayloadParameters,
                nonMandatoryRequestPayloadParameters = st.session_state.nonMandatoryRequestPayloadParameters,
                mandatoryResponsePayloadParameters = st.session_state.mandatoryResponsePayloadParameters,
                nonMandatoryResponsePayloadParameters = st.session_state.nonMandatoryResponsePayloadParameters
        )



            if model == 'GPT-3.5 Turbo':

                st.write('Using: '+model)

                llm = OpenAI(model_name= "gpt-3.5-turbo-0613", temperature = 0.5)

                if(len(fprompt) != 0):
                    response = llm(fprompt)
                    st.code(response)
                    st.session_state['response'] = response
                
                if(len(response) != 0):
                    resultStatus = False

            if model ==  'Google Gemini Pro': 
                st.write('Using: ' + model)

                llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)

                if(len(fprompt) != 0):
                    response = llm.invoke(fprompt)
                    st.code(response.content)
                    st.session_state['response'] = response.content

            if model ==  'GPT-4': 
                st.write('Using: ' + model)

                llm = ChatOpenAI(model_name= "gpt-4o", temperature = 0.5)

                if(len(fprompt) != 0):
                    response = llm.invoke(fprompt)
                    st.code(response.content)
                    st.session_state['response'] = response.content

                if(len(response.content) != 0):
                    resultStatus = False

            if model == None:
                st.error('Please Select a LLM')

        st.divider()
        st.subheader('Write Test Scripts')
        st.caption(':red[Before selecting the language, please copy the required test cases to the clipboard or note them down to write the script out of the generated test cases before clicking the button.]')

        st.selectbox('Select the Test Script Language',('Python', 'Java', 'Cucumber (Feature File Already Available)','Cucumber (Generate Both Feature File and Script)'),key = 'script_lang',index=0,disabled=resultStatus)
        scripted = st.form_submit_button("Write Test Scripts", disabled=resultStatus)

        if scripted:
            if st.session_state.script_lang == 'Cucumber (Feature File Already Available)' :
                switch_page("BDD With Feature File")
            
            if st.session_state.script_lang == 'Cucumber (Generate Both Feature File and Script)' :
                switch_page("BDD Without Feature File")

            if st.session_state.script_lang == 'Java' :
                switch_page("API Test Script Gen Java")

            if st.session_state.script_lang == 'Python' :
                switch_page("API Test Script Gen Python")
            
            else :
                switch_page("API Test Script Gen")







