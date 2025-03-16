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


st.title('Generate Test Scripts for Functional UI Testing')

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

# template = """ 
# I want to generate selenium test script for below test case. Below I mentioned the test case and business requirement.\n
# Test case type - {testCaseType}\n
# Test case - {testCase}\n
# Business Requirement:\n
# User Story Name - {userStoryName}\n
# Main Business Functionality - {mainBusinessFunc}\n
# Sub Business Functionalites - {subBusinessFunc}\n
# Precondition - {precondition}\n
# Type of End Users - {endUsersType}\n
# Think you are a QA engineer. You need to mainly consider above mention test case and generate selenium script for {language} according to that testcase, as a professional QA engineer. When you write selenium script please follow coding best practices, coding standards, exception handling as a QA engineer.
# """

template = """ 
Test case type - {testCaseType}\n
Test case - {testCase}\n
Business Requirement:\n
User Story Name - {userStoryName}\n
Main Business Functionality - {mainBusinessFunc}\n
Sub Business Functionalites - {subBusinessFunc}\n
Precondition - {precondition}\n
Additional Information - {additionalInfo}
Already available functions : - {reusableFunc},(please reuse these where applicable and these are the only existing functions.)\n
Instructions - Develop a Selenium script in {language} to automate the positive test case described above.
Adhere to coding best practices, coding standards, and implement robust exception handling.
Use the web driver as defined.
Consider parameterization for flexibility and adaptability.
Implement detailed logging for debugging purposes.
Utilize a testing framework such as TestNG or JUnit.
Follow the Page Object Model for improved maintainability.
Include waits effectively using explicit waits.
Don't be limited to the already available functions.
Don't assume any other function is available than the mentioned ones.
When generating a new function other than the already available ones, always assume the existence of appropriate UI components and provide the code in the body.
Only for the given already implemented functions, You are not required to write the code in the body. 
If any already existing functions are reused, indicate where they are used.
Only answer me with the code and nothing else. don't give additional text with answer. I need code only as answer.

"""


# template = """ 
# I want to generate selenium test script for below test case. Below I mentioned the test case and business requirement.\n
# Test case type - {testCaseType}\n
# Test case - {testCase}\n
# Business Requirement:\n
# User Story Name - {userStoryName}\n
# Main Business Functionality - {mainBusinessFunc}\n
# Sub Business Functionalites - {subBusinessFunc}\n
# Precondition - {precondition}\n
# Type of End Users - {endUsersType}\n
# Additional Information - {additionalInfo}
# Already available functions (please use if applicable and these are only the existing functions): : {reusableFunc}n
# Instructions:  Develop a Selenium script in {language} to automate the positive test case described above.
# Adhere to coding best practices, coding standards, and implement robust exception handling.
# Use the web driver as defined.
# {instructions}
# Include waits effectively using explicit waits.
# Not required to write already implemented functions.
# Do not keep functions with only comments, assume the existence of appropriate UI components and implement the code as much as you can.
# If any existing functions are reused, indicate where they are used.



# """

additional_info_text: str = """
Element identification methods: XPaths\n
Programming language to use: Java with Selenium\n
Web driver to use: Chrome
"""


instruction_text: str = """
Consider parameterization for flexibility and adaptability.
Implement detailed logging for debugging purposes.
Utilize a testing framework such as TestNG or JUnit.
Follow the Page Object Model for improved maintainability
"""

on = st.toggle('Populate fields with a sample scenario')

if not on:

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx']) 

    def get_excel_file_content_as_binary(file_path):
        with open(file_path, "rb") as file:
            return file.read()

    file_path = 'template_files/Selenium_Code_Template.xlsx'

    excel_file_content = get_excel_file_content_as_binary(file_path)

    st.download_button(label="Download Selenium Code Template",
                    data=excel_file_content,
                    file_name="Selenium_Code_Template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    uploaded_file = None

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        data = df.iloc[0]  # Assuming data for form defaults is in the first row

        with st.form('api_ts_gen'):
            st.text_input('Test Case Type', value=data.iloc[0], placeholder='Enter Test Case Type', key='testCaseType', help="Enter the type of the test case here. Ex: Positive, Negative etc.")
            st.text_input('Test Case', value=data.iloc[1], placeholder='Please Type the Test Case', key='testCase', help="Please Enter the Test Case to be tested here.")
            st.text_input('User Story Name', value=data.iloc[2], placeholder='Enter the User Story Name', key='userStoryName', help="Please Enter the Name of the User Story here")
            st.text_area('Main Business Functionality', value=data.iloc[3], placeholder='', key='mainBusinessFunc', help="Enter the Primary Business Functionality to be tested.")
            st.text_area('Sub Business Functionalities', value=data.iloc[4], placeholder='', key='subBusinessFunc', help="Enter the Sub business functionalities to be tested.")
            # If there's a value for Test Scenario Combination, adjust data.iloc[] accordingly and uncomment:
            # st.text_input('Test Scenario Combination', value=data.iloc[5], placeholder='', key='testScenarioCombination')
            st.text_input('Precondition', value=data.iloc[5], placeholder='Precondition', key='precondition', help="Please Enter the Pre Conditions that should be met.")
            st.text_input('Type of End Users', value=data.iloc[6], placeholder='Type of End Users', key='endUsersType', help="Enter the type of the End Users as per their roles.")
            st.text_area('Additional Information', value=data.iloc[7], placeholder='Element Identification methods, Web drivers to be used etc.', key='additionalInfo', help='Information such as Element Identification Methods Ex: XPath, Web Drivers Ex: Chrome etc.')
            st.text_area('Reusable Functions', value=data.iloc[8], placeholder='Functions that are readily available to be reused', key='reusableFunc', help='Please type the definitions of the readily available functions that should be reused in the script.')
            # If additional instructions are provided in the Excel, adjust data.iloc[] accordingly and uncomment:
            # st.text_area('Instructions', value=data.iloc[9], placeholder='Any additional instructions to the model to be followed when generating the script', key='instructions', help='Please add any instructions that the LLM should follow when generating the script.')
            language = data.iloc[9]
            language_with_lower_case = language.lower()

            if language_with_lower_case == 'python':
                st.selectbox('Required Language', options=['Python', 'Java'], index=0, key='language')
            elif language_with_lower_case == 'java':
                st.selectbox('Required Language', options=['Python', 'Java'], index=1, key='language')
            else:
                st.selectbox('Required Language', options=['Python', 'Java'], index=1, key='language')
            #st.selectbox('Select the Language',('Python', 'Java'),key = 'language',placeholder='Select for which language the selenium script should be generated',index=1)


            submitted = st.form_submit_button("Generate")
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")


elif on:
    with st.form('api_ts_gen'):
        st.text_input('Test Case Type', placeholder='Enter Test Case Type',value='Positive Test Case ', key = 'testCaseType',help="Enter the type of the test case here. Ex: Positive, Negative etc.")
        st.text_input('Test Case', placeholder='Please Type the Test Case', value="Verify that the meeting details are added to the agent's timeline and calendar option when a meeting request is received from XYZ end.",key = 'testCase', help="Please Enter the Test Case to be tested here.")
        st.text_input('User Story Name', placeholder='Enter the User Story Name', value='Agent and Gold Customer Meeting Scheduling in the ABC Application',key = 'userStoryName',help="Please Enter the Name of the User Story here")
        st.text_area('Main Business Functionality', placeholder='', value='The meeting details are added to the relevant agent timeline and calendar, The meeting details are added to the relevant customer calendar and dashboard, The agent can initiate the meeting in ABC Application and the customer should be able to join the meeting.',key = 'mainBusinessFunc',help="Enter the Primary Business Functionality to be tested.")
        st.text_area('Sub Business Functionalities', placeholder='', value='N/A',key = 'subBusinessFunc',help="Enter the Sub business functionalities to be tested.")
        # st.text_input('Test Scenario Combination', placeholder='', key = 'testScenarioCombination')
        st.text_input('Precondition', placeholder='Precondition', value='The meeting request should come from the XYZ end. ',key = 'precondition', help="Please Enter the Pre Conditions that should be met.")
        st.text_input('Type of End Users', placeholder='Type of End Users',value='Two types of agents named Relationship Managers and Financial Advisors', key = 'endUsersType', help="Enter the type of the End Users as per their roles.")
        st.text_area('Additional Information',placeholder='Element Identification methods, Web drivers to be used etc.',value='Element identification methods: XPaths\nWeb driver to use: Chrome', key='additionalInfo', help='Information such as Element Identification Methods Ex: XPath, Web Drivers Ex: Chrome etc.' )
        st.text_area('Reusable Functions', placeholder='Functions that are readily available to be reused',key ='reusableFunc',value='login(driver, username, password), goToScheduledMeetingPage(driver), removeUserFromTheMeeting(driver, userId)' ,help='Please type thed definitions of the readily available functions that should be reused in the script.',height=75)
        #st.text_area('Instructions', placeholder='Any additional instructions to the model to be followed when generating the script',value='Consider parameterization for flexibility and adaptability.\nImplement detailed logging for debugging purposes.\nUtilize a testing framework such as TestNG or JUnit.\nFollow the Page Object Model for improved maintainability.', key='instructions',help='Please add any instructions that the LLM should follow when generating the script.',height=150)
        st.selectbox('Select the Language',('Python', 'Java'),key = 'language',placeholder='Select for which language the selenium script should be generated',index=1)
        submitted = st.form_submit_button("Generate")

else: 
    with st.form('api_ts_gen'):
        st.text_input('Test Case Type', placeholder='Enter Test Case Type', key = 'testCaseType',help="Enter the type of the test case here. Ex: Positive, Negative etc.")
        st.text_input('Test Case', placeholder='Please Type the Test Case', key = 'testCase', help="Please Enter the Test Case to be tested here.")
        st.text_input('User Story Name', placeholder='Enter the User Story Name', key = 'userStoryName',help="Please Enter the Name of the User Story here")
        st.text_area('Main Business Functionality', placeholder='', key = 'mainBusinessFunc',help="Enter the Primary Business Functionality to be tested.")
        st.text_area('Sub Business Functionalities', placeholder='', key = 'subBusinessFunc',help="Enter the Sub business functionalities to be tested.")
        # st.text_input('Test Scenario Combination', placeholder='', key = 'testScenarioCombination')
        st.text_input('Precondition', placeholder='Precondition', key = 'precondition', help="Please Enter the Pre Conditions that should be met.")
        st.text_input('Type of End Users', placeholder='Type of End Users', key = 'endUsersType', help="Enter the type of the End Users as per their roles.")
        st.text_area('Additional Information',placeholder='Element Identification methods, Web drivers to be used etc.',key='additionalInfo', help='Information such as Element Identification Methods Ex: XPath, Web Drivers Ex: Chrome etc.' )
        st.text_area('Reusable Functions', placeholder='Functions that are readily available to be reused',key ='reusableFunc',help='Please type thed definitions of the readily available functions that should be reused in the script.')
        #st.text_area('Instructions', placeholder='Any additional instructions to the model to be followed when generating the script', key='instructions',help='Please add any instructions that the LLM should follow when generating the script.')
        st.selectbox('Select the Language',('Python', 'Java'),key = 'language',placeholder='Select for which language the selenium script should be generated',index=None)
        submitted = st.form_submit_button("Generate")




if submitted: 
    ui_ts_template = PromptTemplate.from_template(template)
    ui_ts_template.input_variables = ['testCaseType','testCase','userStoryName','mainBusinessFunc','subBusinessFunc','precondition','language','additionalInfo','reusableFunc']

    formatted_prompt = ui_ts_template.format(
        testCaseType = st.session_state.testCaseType,
        testCase = st.session_state.testCase,
        userStoryName = st.session_state.userStoryName,
        mainBusinessFunc = st.session_state.mainBusinessFunc,
        subBusinessFunc = st.session_state.subBusinessFunc,
        # testScenarioCombination = st.session_state.testScenarioCombination,
        precondition = st.session_state.precondition,
        #endUsersType = st.session_state.endUsersType,
        language = st.session_state.language,
        additionalInfo = st.session_state.additionalInfo,
        reusableFunc = st.session_state.reusableFunc,
        #instructions = st.session_state.instructions

    )


    if model == 'GPT-3.5 Turbo':

        st.write('Using: '+model)

        llm = OpenAI(model_name= "gpt-3.5-turbo-0613", temperature = 0.1)

        if(len(formatted_prompt) != 0):
            response = llm(formatted_prompt)
            st.session_state['response_code'] = response

                    

    if model ==  'GPT-4': 
        st.write('Using: ' + model)

        llm = ChatOpenAI(model_name= "gpt-4o", temperature = 0, model_kwargs={"seed": 10})

        if(len(formatted_prompt) != 0):
            response = llm.invoke(formatted_prompt)
            st.session_state['response_code'] = response.content
            

    if model ==  'Google Gemini Pro': 
        st.write('Using: ' + model)

        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key = GOOGLE_API_KEY)

        if(len(formatted_prompt) != 0):
            response = llm.invoke(formatted_prompt)
            st.session_state['response_code'] = response.content

    if model == None:
        st.error('Please Select a LLM')


def describe_code_function(response, llm):

    full_prompt = f"""
        Generated Code : {response}

        Think you as the Expert Automation Quality Assurance Engineer, Understand above given selenium code. 
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

    if st.session_state.language == 'Python':
        file_extension = 'py'
    else:
        file_extension = 'java'

    filename = f"selenium_code.{file_extension}"
    
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




