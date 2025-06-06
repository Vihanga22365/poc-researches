import os
import sqlite3
# Conceptual Code: Hierarchical Research Task
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from dotenv import load_dotenv
from typing import List, Dict, Any
# from google.adk.tools.mcp_tool import MCPToolset, StdioServerParameters
# from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import requests
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from google.adk.models.lite_llm import LiteLlm

llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def format_docs(docs):
  format_D="\n\n".join([d.page_content for d in docs])
  return format_D


def rag_inquiry_handler_tool(question: str) -> str:
    """
      Ask a question using the RAG.
      User can ask a question about our company and get the answer from the RAG.
    
      Args:
          question (str): The question to ask
          
      Returns:
          str: The answer to the question
        
           
    """
    docsearch = FAISS.load_local(os.path.join(os.path.dirname(__file__), "./faiss_db/"), embeddings, allow_dangerous_deserialization=True)
    retriever = docsearch.as_retriever(search_kwargs={"k": 5})
    
    template = """
            You are report generator.
            In the 'Question' section include the question.
            In the 'Context' section include the nessary context to generate the section of the report.
            According to the 'Context', please generate the section of the report.
            Use only the below given 'Context' to generate the section of the report.
            Make sure to provide the answer from the 'Context' only.
            Provide answer only. Nothing else.
            Context: {context}
            Question : {question}
            """
              
    prompt=ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(question)
    return response

def code_interpreter_tool(code: str) -> str:
    """
    Executes the provided Python code and returns the output.
    """
    # Define a restricted execution environment
    restricted_globals = {"__builtins__": {}}
    restricted_locals = {}

    try:
        # Execute the code within the restricted environment
        exec(code, restricted_globals, restricted_locals)
        return str(restricted_locals)
    except Exception as e:
        return f"Error: {e}"
    
def get_user_interactions(user_identifier: str) -> List[Dict[str, Any]]:
    """
    Retrieve all interactions for a specific user by ID or partial name.
    
    Args:
        user_identifier (str): The ID or partial name of the user to retrieve interactions for
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing interaction data
        Each dictionary contains: ID, UserID, Channel, and Interaction
    """
    try:
        conn = sqlite3.connect("customer_data.db")
        cursor = conn.cursor()
        
        # Query to get all interactions for the specified user
        query = """
        SELECT ui.ID, ui.UserID, ui.Channel, ui.Interaction 
        FROM user_interactions ui
        JOIN user u ON ui.UserID = u.UserID
        WHERE ui.UserID = ? OR u.Name LIKE ?
        ORDER BY ui.ID
        """
        
        # Add wildcards for partial name matching
        name_pattern = f"%{user_identifier}%"
        cursor.execute(query, (user_identifier, name_pattern))
        interactions = cursor.fetchall()
        
        # Convert the results to a list of dictionaries
        result = []
        for interaction in interactions:
            result.append({
                'ID': interaction[0],
                'UserID': interaction[1],
                'Channel': interaction[2],
                'Interaction': interaction[3]
            })
        
        return result
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_user_transactions(user_identifier: str) -> List[Dict[str, Any]]:
    """
    Retrieve all transactions for a specific user by ID or partial name.
    
    Args:
        user_identifier (str): The ID or partial name of the user to retrieve transactions for
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing transaction data
        Each dictionary contains: ID, UserID, Type, Amount, Date, Counterparty, 
        AccountNumber, and Description
    """
    try:
        conn = sqlite3.connect("customer_data.db")
        cursor = conn.cursor()

        print("Connecting to the database to retrieve user transactions...")
        print(f"User Identifier: {user_identifier}")
        print("Executing query to fetch transactions...")
        
        # Query to get all transactions for the specified user
        query = """
        SELECT ut.ID, ut.UserID, ut.Type, ut.Amount, ut.Date, ut.Counterparty, ut.AccountNumber, ut.Description 
        FROM user_transactions ut
        JOIN user u ON ut.UserID = u.UserID
        WHERE ut.UserID = ? OR u.Name LIKE ?
        ORDER BY ut.Date
        """
        
        # Add wildcards for partial name matching
        name_pattern = f"%{user_identifier}%"
        cursor.execute(query, (user_identifier, name_pattern))
        transactions = cursor.fetchall()
        
        # Convert the results to a list of dictionaries
        result = []
        for transaction in transactions:
            result.append({
                'ID': transaction[0],
                'UserID': transaction[1],
                'Type': transaction[2],
                'Amount': transaction[3],
                'Date': transaction[4],
                'Counterparty': transaction[5],
                'AccountNumber': transaction[6],
                'Description': transaction[7]
            })
        
        return result
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    finally:
        if conn:
            conn.close()
    
    
# def online_web_search(search_query: str):
#     """
#     Perform an online web search.
    
#     Parameters:
#         search_query (str): The search query.
    
#     Returns:
#         dict: The search results in JSON format.
#     """
#     url = "https://api.tavily.com/search"
#     headers = {"Content-Type": "application/json"}
#     data = {"api_key": "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", "query": search_query, "max_results": 10}
    
#     response = requests.post(url, json=data, headers=headers)
#     return response.json()

def online_web_search_openai_tool(search_query: str):
    """
    Perform an online web search.
    
    Parameters:
        search_query (str): The search query.
    
    Returns:
        dict: The search results in JSON format.
    """
    llm = ChatOpenAI(model="gpt-4o-mini")

    tool = {"type": "web_search_preview"}
    llm_with_tools = llm.bind_tools([tool])

    response = llm_with_tools.invoke(search_query)
    return response 

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import io
import base64
import time

def forecast_monthly_deposite_or_expense(json_data: dict, chart_type: str = "deposit", forecast_months: int = 6) -> tuple:
    """
    Generates a monthly deposite/expense forecast plot and a table of forecasted values
    for the specified number of months from a given JSON financial transaction data.

    Args:
        json_data (dict): A dictionary containing transaction data in the format:
                          {"transactions": [{"ID": ..., "Type": "CR/DR", "Amount": ..., "Date": ..., ...}]}
        chart_type (str): Type of chart to generate - "deposit" or "expenses"
        forecast_months (int): Number of months to forecast (default: 6)

    Returns:
        tuple: (image_path, forecast_table) - Path to the saved image and forecast table
    """
    warnings.filterwarnings("ignore")

    # Get the absolute path to the static directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "static", "charts")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame
    df = pd.DataFrame(json_data['transactions'])

    # Filter transactions based on chart type
    if chart_type.lower() == "deposit":
        filtered_df = df[df['Type'] == 'CR'].copy()
        title_prefix = "Monthly Deposite"
    else:  # expenses
        filtered_df = df[df['Type'] == 'DR'].copy()
        title_prefix = "Monthly Expenses"

    # Convert 'Date' to datetime objects
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    # Set 'Date' as index
    filtered_df.set_index('Date', inplace=True)

    # Resample to monthly frequency and sum the 'Amount'
    monthly_data = filtered_df['Amount'].resample('MS').sum()

    # Convert to DataFrame for easier plotting and manipulation
    monthly_df = monthly_data.reset_index()
    monthly_df.rename(columns={'Date': 'ds', 'Amount': 'y'}, inplace=True)
    monthly_df['ds'] = monthly_df['ds'].dt.to_period('M')
    monthly_df = monthly_df.set_index('ds')

    # Define SARIMAX parameters (simplified for small datasets)
    order = (1, 0, 0)
    seasonal_order = (0, 0, 0, 0)

    # Fit SARIMAX Model
    model = SARIMAX(monthly_df['y'], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast for the specified number of months
    forecast_result = model_fit.get_forecast(steps=forecast_months)
    forecast_mean = forecast_result.predicted_mean
    forecast_conf_int = forecast_result.conf_int()

    # Create dates for the forecast period
    last_date_period = monthly_df.index.max()
    forecast_period_index = pd.period_range(start=last_date_period + 1, periods=forecast_months, freq='M')

    # Prepare forecast data for plotting
    forecast_df = pd.DataFrame({
        'yhat': forecast_mean.values,
        'yhat_lower': forecast_conf_int['lower y'].values,
        'yhat_upper': forecast_conf_int['upper y'].values
    }, index=forecast_period_index)

    # --- Monthly Forecast Plot ---
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(monthly_df.index.to_timestamp(), monthly_df['y'], 
             label=f'Historical {title_prefix}', marker='o', color='blue')
    
    # Plot forecast data
    plt.plot(forecast_df.index.to_timestamp(), forecast_df['yhat'], 
             label=f'Forecasted {title_prefix} ({forecast_months} months)', 
             color='red', linestyle='--', marker='x')
    
    # Add confidence interval
    plt.fill_between(forecast_df.index.to_timestamp(), 
                     forecast_df['yhat_lower'], forecast_df['yhat_upper'], 
                     color='pink', alpha=0.3, 
                     label=f'95% Confidence Interval ({forecast_months} months)')
    
    # Add vertical line to separate historical and forecast data
    last_historical_date = monthly_df.index.to_timestamp()[-1]
    plt.axvline(x=last_historical_date, color='gray', linestyle='--', alpha=0.5,
                label='Forecast Start')
    
    plt.title(f'{title_prefix} Forecast (SARIMAX Model) - Next {forecast_months} Months', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Total Amount', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Generate a unique filename with timestamp to prevent caching
    timestamp = int(time.time())
    image_filename = f'monthly_forecast_{timestamp}.png'
    image_path = os.path.join(output_dir, image_filename)
    
    # Save plot to file
    plt.savefig(image_path, format='png', bbox_inches='tight')
    plt.close()

    # --- Forecasted Values Table ---
    forecast_output = forecast_df[['yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_output.index = forecast_output.index.strftime('%Y-%m')
    forecast_table = forecast_output.to_markdown(numalign="left", stralign="left")

    # Return the relative path for the image URL
    relative_image_path = f'static/charts/{image_filename}'
    return relative_image_path, forecast_table

# Mid-level agent combining tools
customer_information_agent = LlmAgent(
    name="CustomerInformationAgent",
    model=LiteLlm(model="openai/gpt-4.1"),
    description="""
        **Tools**
            - 'get_user_interactions' - Get the interactions of the user.
            - 'get_user_transactions' - Get the transactions of the user.

        **Objective**
            - You are a Customer Information Agent and your goal is to assist users in finding information about their interactions and transactions.
            - Clearly understand the 'User Query' and use the appropriate tool to get the required information.

        **Instructions**
            - If you want to get the interactions of the user, pass the 'UserID' to the tool 'get_user_interactions' and get the interactions of the user.
            - If you want to get the transactions of the user, pass the 'UserID' to the tool 'get_user_transactions' and get the transactions of the user.
            - If you can't find the required information, using the available tools, mention it in final answer. 
            - If you can't find the required information, don't answer those by yourself, mention it in final answer.

        **Expected Output**
            - Give the answer with JSON format.
    """,
    tools=[get_user_interactions, get_user_transactions]
)

deposite_or_expense_prediction_agent = LlmAgent(
    name="DepositeOrExpensePredictionAgent",
    model=LiteLlm(model="openai/gpt-4.1"),
    description="""
        **Tools**
            - 'forecast_monthly_deposite_or_expense' - Forecast the monthly deposite or expense for the requested months.

        **Objective**
            - You are a deposite or expense Prediction Agent and your goal is to assist users in forecasting their monthly deposite or expense.
            - According to user's previous transactions, forecast the monthly deposite or expense for the requested months.

        **Instructions**
            - If you want to forecast the monthly deposite or expense for the requested months, Make sure to identify below given information from the user query and pass it to the tool 'forecast_monthly_deposite_or_expense' to get the forecasted deposite or expense.,
                - JSON data - JSON data of the transactions of the user.
                - Chart Type - Type of chart to generate - "deposit" or "expenses"
                - Forcast Months - Number of months to forecast
            - If you can't find the required information, using the available tools, mention it in final answer. 
            - If you can't find the required information, don't answer those by yourself, mention it in final answer.

        **Expected Output**
            - Chart image of forcasted deposite or expense for the next requested months period with 100% width - Image path (Make sure to use absolute attribute in image element same as given. Eg: src and width) = <img src="http://localhost:8090/{image_path}" width="900" alt="Monthly Income Forecast">
            - Table view of forcasted deposite or expense for the next requested months period.
    """,
    tools=[forecast_monthly_deposite_or_expense]
)

# High-level agent delegating research
root_agent = LlmAgent(
    name="ManagerAgent",
    model=LiteLlm(model="openai/gpt-4.1"),
    instruction="""
        
        **Tools** 
            - 'rag_inquiry_handler_tool' - Make sure only search and gather the required information about the our company 'JPMorgan Chase & Co'.
            - 'online_web_search_openai_tool' -  Make sure only search and gather the required information about the user given company, other than our company 'JPMorgan Chase & Co'.
            - 'code_interpreter_tool' - Execute the provided Python code and return the output.
            - 'customer_information_agent' - Get the information about the user's interactions and transactions.
            - 'deposite_or_expense_prediction_agent' - Forecast the monthly deposite / expenses for the next requested months period and show the chart image and table view of forecasted deposite or expense for the next requested months period.
        
        **Objective**
            - You are a Researcher Agent In Financial Company and your goal is to assist users in finding information on the web, in documents, and through calculations.
            - Clearly understand the 'User Query' and use the appropriate tool to get the required information.
            - Generate the answer based on the information gathered.    
            
        **Instructions**
            - If you want to search and gather the required information about the our company 'JPMorgan Chase & Co', use the tool 'rag_inquiry_handler_tool'.
            - If you want to search and gather the required information about the user given company, other than our company 'JPMorgan Chase & Co', use the tool 'online_web_search'.
            - If you want to execute the provided Python code and return the output, use the tool 'code_interpreter_tool'.
            - If you want to get the specific information about the user's interactions and transactions, use the tool 'customer_information_agent'.
            - If you want to forecast the monthly deposite / expenses for the next requested months period, make sure pass transacion JSON to the tool 'deposite_or_expense_prediction_agent'. and show the chart image and table view of forecasted deposite or expense for the next requested months period.
            - When enters the 'UserID' or 'Name' of the user, don' ask for the 'UserID' or 'Name' again. Only pass given 'UserID' or 'Name' to the tool 'customer_information_agent' to get the information about the user.
            - If you cannot find user details from the 'customer_information_agent' ask the user to provide the 'UserID' or 'Name' of the user to get the information or correct the 'UserID' or 'Name' of the user to get the information.
            - If you want to perform any calculations, don't do it manually, generate the python code and use the tool 'code_interpreter_tool' to execute the code and get the output.
            - Think step by step about the 'User Query' and use the appropriate tool to get the required information.
            - If you can't find the required information, using the available tools, mention it in final answer. 
            - If you can't find the required information, don't answer those by yourself, mention it in final answer.
            - Finally, provide the answer to the 'User Query' based on the information gathered.
            - When show the answer, make sure to use human readable format. (Don't show the answer in JSON format) and make sure to don't show summary of the answer, just show the answer as it is, if user doesn't ask for summary.
            
            
            Example:
                If user ask : "What is Director's name of JPMC and Director's name of Meta Company?" (This is related to your company or your company's competitor)
                Gather the information from the 'rag_inquiry_handler_tool' about the Director's name of JPMC.
                Gather the information from the 'online_web_search_openai_tool' about the Director's name of Meta Company.
                
                Output :  
                    **Name of the Director of JPMC**
                        <Director's Name of JPMC>
                    
                    **Name of the Director of Meta Company**
                        <Director's Name of Meta Company>
                        
                        
                If user ask : "Director's name of Meta Company?" (User directly asked about the other company)
                Mention that you can't find the required information.
                Output :
                    I can't assist to get the information about the Director's name of Meta Company. Because, it's not related to the my capabilities. My capabilities are limited to the information about the company 'JPMorgan Chase & Co'.
                               
                        
                If user ask : "What is Director's name of JPMC and Meta Company. after tell me the revenue of JPMC in 2023?" (This is related to your company or your company's competitor)
                Gather the information from the 'rag_inquiry_handler_tool' about the Director's name of JPMC and Gather the information from 'rag_inquiry_handler_tool' for calculating the revenue of JPMC in 2023.
                Gather the information from the 'online_web_search_openai_tool' about the Director's name of Meta Company.
                Generate the python code to calculate the revenue of JPMC using the tool 'code_interpreter_tool'.
                Output :  
                
                    **Name of the Director of JPMC**
                        <Director's Name of JPMC>
                        
                    **Revenue of JPMC in 2023**
                        <Revenue of JPMC in 2023>
                        
                    **Name of the Director of Meta Company**
                        <Director's Name of Meta Company>
                        
                        
                If user ask : "What is the Capital of India?" (This is not related to your company)
                Mention that you can't find the required information.
                Output :  
                    I can't assist to get the information about the capital of India. Because, it's not related to the my capabilities. My capabilities are limited to the information about the company 'JPMorgan Chase & Co'.


                If user ask : "Predict the monthly deposite for the next 6 months  for UserID U001?" 
                Gather the information from the 'customer_information_agent' about the transactions of the user.
                Gather the information from the 'deposite_or_expense_prediction_agent' about the transactions of the user. (Make sure to chart type and forecast months that entered by the user)
                Output :  
                
                    **Chart image of forcasted deposite for the next 6 months** (Make sure to show the chart image with 100% width)
                        <Chart image of forcasted deposite for the next 6 months>  (Make sure to chart type and forecast months that entered by the user)
                        
                    **Table view of forcasted deposite for the next 6 months**
                        <Table view of forcasted deposite for the next 6 months> (Make sure to chart type and forecast months that entered by the user)
    """,
    tools=[agent_tool.AgentTool(agent=customer_information_agent), agent_tool.AgentTool(agent=deposite_or_expense_prediction_agent), rag_inquiry_handler_tool, online_web_search_openai_tool, code_interpreter_tool]
)