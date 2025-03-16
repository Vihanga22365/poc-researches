import streamlit as st
from langchain.tools import tool
import requests
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import time
from crewai import Agent, Task, Crew

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Base URL of your API
BASE_URL = "http://127.0.0.1:8080"

@tool
def receive_user_input(user_input: str) -> dict:
    """
    Sends user input to the /user_input/ endpoint.
    
    Args:
        user_input (str): The user's input to be sent to the API.
    
    Returns:
        dict: The response from the API.
    """
    url = f"{BASE_URL}/user_input/"
    payload = {"response": user_input}
    response = requests.post(url, json=payload)
    time.sleep(6)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to send user input: {response.status_code} - {response.text}")

@tool
def get_question() -> dict:
    """
    Retrieves the latest question asked by the bot from the /get_question/ endpoint.
    
    Returns:
        dict: The latest question and metadata.
    """
    time.sleep(6)
    url = f"{BASE_URL}/get_question/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return {"detail": "No question has been asked yet."}
    else:
        raise Exception(f"Failed to retrieve question: {response.status_code} - {response.text}")
    
user_details = {
    "firstName": "Chathusha",
    "lastName": "Wijenayake",
    "fullName": "Chathusha Wijenayake",
    "email": "chathusha@gmail.com",
    "id": "78965432",
    "customerType": "Cards",
    "billPayees":[
       {
          "payeeName":"Just Energy",
       },
       {
          "payeeName":"AT&T",
       }
    ],
    "creditCardAccounts": [
        {
            "accountId": "1234-4567-8907",
            "accountName": "Credit Card...8907",
            "creditLimit": "10000",
            "availableCreditLimit": "9500",
            "totalBalance": "500",
            "statementDate": "06/20/2024",
            "minimumPayment": "50"
        },
        {
            "accountId": "9087-9876-7864",
            "accountName": "Delta Card...7864",
            "creditLimit": "8500",
            "availableCreditLimit": "6000",
            "totalBalance": "2500",
            "statementDate": "06/20/2024",
            "minimumPayment": "100"
        }
    ]
}    
    
make_payment_prompt = f"""

***Objective***
Think you as a banking assistant. You need to extract essential information from the "user_details" represented as JSON, such as name, email, customer type, and id. You will offer assistance to the user on making a payment based on this data.

***Instructions***
If customer has multiple banking/card accounts, prompt them to choose an account by displaying the last four digits of each account ID. Once an account is selected, get the account ID, account name, credit limit, available credit limit, total balance, statement date, and minimum payment linked to that specific account for task execution.

Don't use greetings like "Hello" or "Hi" every time. Greet the user by their name when appropreate to establish the convincing conversation.

Consider the user's details here: {user_details}

Pay attention to the information mentioned in the user's query - {{utterance}}. If any specifications related to the payment and bill pay are included, utilize that and avoid repetitively asking for the same information.

You need to gather the following details: 
   - Account Number (mandatory): Extract from user_details. If multiple accounts, ask the user to select by showing the last 4 digits of each account ID. If user enter invalid or wrong account number, ask the user to provide the correct account number.
   - Bill Payee (mandatory): Can be extracted from the user's details under billPayees. Show all bill payees to the user and ask to select one. 
   - Pay amount (mandatory): Request this from the user. Confirm the details if provided in the user's input query, and use it for the task upon confirmation.
   - Date need to process the payment (mandatory): Request a date to process the payment.

Approach the user's query sequentially and try to identify the correct "Bill Payee", "Pay amount" and "Date need to process the payment" without redundantly asking the customer. If you manage to extract any details from the user's query, request individual confirmation from the user, and don't ask all the details at once.

Verify the "Pay amount" against the "creditLimit" and "availableCreditLimit" of the chosen account. If the transfer amount surpasses the available credit limit, prompt the user to enter a valid transfer amount.
   
When inquiring the first question, explain the task to the user and make sure all necessary inputs are provided. Continue the query until all mandatory details are collected. 

Use the "ask_human" tool for questioning. 

***Additional guidelines***
Ask the user again only if the reply or answer to any question is confusing and a decision cannot be made from the reply. 

If the user interrupts the task and provide irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
    
"""


evaluate_propmt = f"""You are a RELEVANCE grader; providing the relevance of the given CHAT_GUIDANCE to the given CHATBOT_TRANSCRIPT.
            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

            A few additional scoring guidelines:

            - Long CHATBOT_TRANSCRIPT should score equally well as short CHATBOT_TRANSCRIPT.

            - RELEVANCE score should increase as the CHATBOT_TRANSCRIPT provides more RELEVANT context to the CHAT_GUIDANCE.

            - RELEVANCE score should increase as the CHATBOT_TRANSCRIPT provides RELEVANT context to more parts of the CHAT_GUIDANCE.

            - CHATBOT_TRANSCRIPT that is RELEVANT to some of the CHAT_GUIDANCE should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

            - CHATBOT_TRANSCRIPT that is RELEVANT to most of the CHAT_GUIDANCE should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

            - CHATBOT_TRANSCRIPT that is RELEVANT to the entire CHAT_GUIDANCE should get a score of 9 or 10. Higher score indicates more RELEVANCE.

            - CHATBOT_TRANSCRIPT must be relevant and helpful for answering the entire CHAT_GUIDANCE to get a score of 10.

            - Never elaborate.
            
                CHAT_GUIDANCE: {make_payment_prompt}

                CHATBOT_TRANSCRIPT: <<previous task output>
                
                Please answer using the entire template below.
                    TEMPLATE: 
                    Score: <The score 0-10 based on the given criteria>
                    Criteria: <Provide the criteria for this evaluation>
                    Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
            """

# Define the testing agent
testing_agent = Agent(
    role="Chatbot Tester",
    goal="Test the chatbot by simulating a conversation and validating its responses.",
    backstory="The chatbot is designed to answer questions from users. Your goal is to test the chatbot's responses.",
    tools=[receive_user_input, get_question],  # Assign the tools to the agent
    verbose=True,  # Enable verbose logging for debugging
    memory=False,
    cache=False
)

# Streamlit App
def main():
    st.set_page_config(page_title="Chatbot Testing Agent ", layout="wide")
    st.title("Chatbot Testing Agent")
    
    # Tone selection dropdown
    tone = st.selectbox(
        "Select a tone for the conversation:",
        ["Friendly and Casual", "Formal and Professional", "Frustrated or Impatient", "Confused or Unclear", 
         "Neutral and Straightforward", "Empathetic or Emotional", "Demanding or Authoritative", 
         "Curious and Inquisitive", "Sarcastic or Ironic", "Polite and Grateful", "Rushed or Hurried"]
    )
    
    # Adjust the task description based on the selected tone
    task_description = f"""
**Task Overview:**
Simulate a conversation with the chatbot to test its functionality for assisting users in making payments. Follow these steps:

**1. Start the Conversation:**
- Begin with: "I need to make a payment."

**2. Main Chatbot Behavior:**
The chatbot is designed to:
- Extract user details (name, email, customer type, ID) from `user_details`.
- If the user has multiple accounts, prompt them to select one by showing the last 4 digits of each account ID.
- Collect the following mandatory details:
  - **Account Number** (mandatory): User must select an account (last 4 digits shown if multiple accounts exist).
  - **Bill Payee** (mandatory): User must select a payee from the `billPayees` list.
  - **Pay Amount** (mandatory): User must provide the amount.
  - **Payment Date** (mandatory): User must provide the date to process the payment.
- Confirm details with the user and avoid redundant questions.
- Handle interruptions politely and confirm whether to terminate or proceed.

**3. Testing Steps:**
- Use the `get_question` tool to retrieve the chatbot's latest question.
- If no question is available, the tool returns: `{{"detail": "No question has been asked yet."}}`.
- Respond to the question using the `receive_user_input` tool.
- Ensure responses are relevant and reflect a **{tone}** tone.
- Repeat the process until the conversation ends.

**4. Rules:**
- Do not repeat questions or responses. Always use `get_question` for the next question.
- Keep responses relevant to the chatbot's questions.
- When the chatbot asking to select the **Bill Payee**, ask "Who are the other bill payees available?" using the `receive_user_input` tool.
- Do not stop the conversation prematurely. Continue until the chatbot completes the flow.
- Refer to the "Main Chatbot Behavior" section for context.

**5. Goal:**
- Validate that the chatbot responds correctly to user input.
- Provide a report on the chatbot's performance after the conversation ends.
"""
    
    # Define the task for the agent
    test_chatbot_task = Task(
        description=task_description,
        agent=testing_agent,  # Assign the testing agent to this task
        expected_output="""
            Chat full transcript between chatbot and you(User)
                TEMPLATE:
                    Chatbot : <Chatbot's question>
                    User : <User response>
        """,
        tools=[receive_user_input, get_question]
    )
    
    
    evaluate_task = Task(
        description=evaluate_propmt,
        agent=testing_agent,  # Assign the testing agent to this task
        expected_output="""
        TEMPLATE: 
            Score: <The score 0-10 based on the given criteria>
            Criteria: <Provide the criteria for this evaluation>
            Supporting Evidence: <Provide your reasons for scoring based on the listed criteria step by step. Tie it back to the evaluation being completed.>
        
        """,
        tools=[]
    )
    
    # Create a crew with the testing agent and task
    crew = Crew(
        agents=[testing_agent],
        tasks=[test_chatbot_task, evaluate_task],
        memory=False,
        cache=False
    )
    
    import re
    
    # Button to start the conversation
    if st.button("Start Conversation"):
        st.write("Starting the conversation...")
        
        # Execute the crew
        result = crew.kickoff()
        
        # Display the result
        st.write("### Conversation Result")
        result_text = str(result.raw)
    
        score_match = re.search(r"(Score:\s*.*?)(?=Criteria|$)", result_text, re.DOTALL)
        criteria_match = re.search(r"(Criteria:\s*.*?)(?=Supporting Evidence|$)", result_text, re.DOTALL)
        evidence_match = re.search(r"(Supporting Evidence:\s*.*)", result_text, re.DOTALL)

        score_text = score_match.group(1).strip() if score_match else "Score: N/A"
        criteria_text = criteria_match.group(1).strip() if criteria_match else "Criteria: N/A"
        evidence_text = evidence_match.group(1).strip() if evidence_match else "Supporting Evidence: N/A"

        st.write(score_text)
        st.write(criteria_text)
        st.write(evidence_text)

# Run the Streamlit app
if __name__ == "__main__":
    main()