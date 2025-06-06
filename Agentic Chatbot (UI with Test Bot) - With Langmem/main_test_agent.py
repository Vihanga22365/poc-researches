import streamlit as st
from crewai.tools import tool
import requests
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import time
from crewai import Agent, Task, Crew
import uuid
from langchain_core.prompts import PromptTemplate
from langmem import create_prompt_optimizer
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore
from langchain.schema import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import difflib
from html import escape

def get_postgres_store() -> AsyncPostgresStore:
    async def initialize():
        store_cm = AsyncPostgresStore.from_conn_string("postgresql://postgres:12345@localhost:5432/langmem")
        store = await store_cm.__aenter__()
        await store.setup()  # Runs migrations to setup tables
        return store
    return asyncio.run(initialize())

if "store" not in st.session_state:
    st.session_state.store = get_postgres_store()
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["LANGSMITH_TRACING"]='true'
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]="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"
os.environ["LANGSMITH_PROJECT"]="Langmem Procedural Memory - Agentic Chat"

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)

# Global variable for evaluation details
evaluation_details = ""

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






store = st.session_state.store
checkpointer = st.session_state.checkpointer
thread_id = st.session_state.thread_id

user_id = "user151"
instructions_id = "instructions_makePayment_"+user_id
instructions_key = "agent_instructions_makePayment_"+user_id

config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

@tool
def optimize_prompt(chatbot_history: str, evaluation_details: str) -> str:
    """
        If user enters user's name, age, or etc, this function can be used to optimize the prompt.

        Args:
            chatbot_history (str): The chatbot's previous conversation with the user.
            evaluation_details (str): The evaluation details.
        Returns:
            optimized_prompt (str): The optimized prompt.
    """
    # Example optimization: Add a prefix to the prompt
    
    history_full = chatbot_history
    evaluation_details = evaluation_details
    

    optimizer = create_prompt_optimizer("openai:gpt-4o", kind="prompt_memory")
    async def async_optimize():
        temp_store_cm = AsyncPostgresStore.from_conn_string("postgresql://postgres:12345@localhost:5432/langmem")
        async with temp_store_cm as temp_store:
            await temp_store.setup()
            item = await temp_store.aget((instructions_id,), key=instructions_key)
            current_prompt = item.value["prompt"]
            print(current_prompt)
            updated_feedback = f""" 
                Chat History: {chatbot_history}
                Evaluation Details: {evaluation_details}
                
                ## Instructions
                    1. Read `Chat History` and `Evaluation Details` very carefully.
                    2. When you read `Chat History`, if user wants to change the insructions from next time, change the instructions accordingly.
                    3. When you read `Chat History`, if user wants to change the structure of the instructions from next time, change the instructions accordingly.
                    4. Make sure to mention exactly what you are changing in the instructions.
                    5. Don't fully change the instructions. Identify the exact part of the instructions that needs to be changed and change only that part.
                    6. Identify below given criteria and examples, and accordingly change the exact part of the instructions that needs to be changed and change only that part.

                            Example: In the `Chat History`, when chatbot asks question, and user not responding directly, and asking for same question differenct format, make sure you can optimize the prompt to avoid asking same question again.
                      
                                1. When user ask content with different format, change the prompt accordingly.
                                    Chatbot : This is your last 5 transactions. Select one of the following:
                                        1. Adidas 2025-01-01 $500
                                        2. Nike 2025-01-02 $600
                                        3. Puma 2025-01-03 $700  
                                        4. Reebok 2025-01-04 $800
                                        5. Under Armour 2025-01-05 $900
                                    User : No need to show all details. Just show me the merchant name only.
                                    Chatbot : This is your last 5 transactions. Select one of the following:
                                        1. Adidas
                                        2. Nike
                                        3. Puma
                                        4. Reebok
                                        5. Under Armour

                                2. When user ask additional information when chatbot asking question, change the prompt accordingly. Update prompt to avoid asking same question again next time.
                                    Chatbot : This is your last 5 transactions. Select one of the following:
                                        1. Adidas 2025-01-01 $500
                                        2. Nike 2025-01-02 $600
                                        3. Puma 2025-01-03 $700
                                        4. Reebok 2025-01-04 $800
                                        5. Under Armour 2025-01-05 $900

                                    User : Please show me with transaction time also.
                                    Chatbot : This is your last 5 transactions. Select one of the following:
                                        1. Adidas 2025-01-01 11:58:00 $500
                                        2. Nike 2025-01-02 12:00:00 $600
                                        3. Puma 2025-01-03 12:01:00 $700
                                        4. Reebok 2025-01-04 12:02:00 $800
                                        5. Under Armour 2025-01-05 12:03:00 $900

                                3. If user request next time when he log in to the system, make sure to ask the same thing, change the prompt accordingly.
                                    Chatbot : This is your last 5 transactions. Select one of the following:
                                        1. Adidas 2025-01-01 $500
                                        2. Nike 2025-01-02 $600
                                        3. Puma 2025-01-03 $700  
                                        4. Reebok 2025-01-04 $800
                                        5. Under Armour 2025-01-05 $900
                                    User : No need to show all details. Just show me the merchant name only. Next time when i log in to the system, make sure to ask the same thing.
                                    Chatbot : This is your last 5 transactions. Select one of the following:
                                        1. Adidas
                                        2. Nike
                                        3. Puma
                                        4. Reebok
                                        5. Under Armour
                     
                                       
            """
            feedback_payload = {"request": updated_feedback}
    
            optimizer_result = optimizer.invoke({"prompt": current_prompt, "trajectories": [(history_full, feedback_payload)]})
            await temp_store.aput((instructions_id,), key=instructions_key, value={"prompt": optimizer_result})
    
    asyncio.run(async_optimize())

def get_diff_html(a, b):
    differ = difflib.Differ()
    diff = list(differ.compare(a.splitlines(), b.splitlines()))

    result = ""
    for line in diff:
        content = escape(line[2:].strip())

        # Skip hint lines
        if line.startswith("? "):
            continue

        # Handle deletions
        if line.startswith("- "):
            if content:
                result += f"<div style='color:red;'>➖ {content}</div>"
            else:
                result += f"<div style='color:red;'>{content}</div>"

        # Handle additions
        elif line.startswith("+ "):
            if content:
                result += f"<div style='color:green;'>➕ {content}</div>"
            else:
                result += f"<div style='color:green;'>{content}</div>"

        # Unchanged lines
        else:
            result += f"<div>{content}</div>"
    
    return result

    
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

Don't use greetings like "Hello" or "Hi" every time. Greet the user by their name when appropriate to establish the convincing conversation.

Consider the user's details here: {{user_details}}

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

If the user interrupts the task and provides irrelevant information to this task, confirm with them politely whether to terminate the current task or proceed. If the user consents to interrupt the current task, conclude the task execution.
    
"""

async def initialize_namespace():
    # Create a new store connection for initializing the namespace to avoid reusing the shared store
    temp_store_cm = AsyncPostgresStore.from_conn_string("postgresql://postgres:12345@localhost:5432/langmem")
    async with temp_store_cm as temp_store:
        await temp_store.setup()  # ensure migrations are run
        namespace_list = await temp_store.alist_namespaces()  # asynchronous namespace listing using a new connection
        matched_item = next((item for item in namespace_list if item[0] == instructions_id), None)
        if matched_item:
            print(matched_item[0])
        else:
            await temp_store.aput((instructions_id,), key=instructions_key, value={"prompt": make_payment_prompt, "user_id": user_id})
# Run the async function to initialize the namespace
asyncio.run(initialize_namespace())


evaluate_propmt = f"""You are a RELEVANCE grader; providing the relevance of the given CHAT_GUIDANCE to the given CHATBOT_TRANSCRIPT.
            Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

            A few additional scoring guidelines:

            - Think step by step and carefully read CHATBOT_TRANSCRIPT and CHAT_GUIDANCE.

            - Long CHATBOT_TRANSCRIPT should score equally well as short CHATBOT_TRANSCRIPT.

            - RELEVANCE score should increase as the CHATBOT_TRANSCRIPT provides more RELEVANT context to the CHAT_GUIDANCE.

            - RELEVANCE score should increase as the CHATBOT_TRANSCRIPT provides RELEVANT context to more parts of the CHAT_GUIDANCE.

            - CHATBOT_TRANSCRIPT that is RELEVANT to some of the CHAT_GUIDANCE should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

            - CHATBOT_TRANSCRIPT that is RELEVANT to most of the CHAT_GUIDANCE should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

            - CHATBOT_TRANSCRIPT that is RELEVANT to the entire CHAT_GUIDANCE but in the CHATBOT_TRANSCRIPT If user asks next time when he log in to the system, make sure to ask the same thing, get a score of 8 or 9. Higher score indicates more RELEVANCE.

            - CHATBOT_TRANSCRIPT that is RELEVANT to the entire CHAT_GUIDANCE should get a score of 9 or 10. Higher score indicates more RELEVANCE.  (Make sure give score of 10 only if the in CHATBOT_TRANSCRIPT, user not asking next time when he log in to the system, change like this.... etc)

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
    tools=[receive_user_input, get_question, optimize_prompt],  # Assign the tools to the agent
    verbose=True,  # Enable verbose logging for debugging
    memory=False,
    cache=False,
    llm=llm
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
    test_bot_prompt = f"""
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
      When chatbot ask to select the account number, you have to requset avaibale credit limit with the account numbers and mention when i log next time make sure to ask account numbers with credit limit.
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
- Do not stop the conversation prematurely. Continue until the chatbot completes the flow.
- Refer to the "Main Chatbot Behavior" section for context.
- After completing the task and when asking 'Do you want to make another task?', Say "No, Thank You" and end the conversation. Make sure don't say anything else after that. 

**5. Goal:**
- Validate that the chatbot responds correctly to user input.
- Provide a report on the chatbot's performance after the conversation ends.
"""
    

    optimization_propmt = f"""
    You are a prompt optimizer. 

        Chatbot's Current Instructions Prompt: {make_payment_prompt}

        Chatbot's History: << User's input and chatbot's response >>

        Evaluation Details: << Evaluation details >>

        **Tools**
        - optimize_prompt: If you think the `Chatbot's Current Instructions Prompt` is not optimized, when you consider the `Chatbot's History` and `Evaluation Details`, you can use this tool to optimize the prompt.

        **Instructions**
            - Read `Chatbot's Current Instructions Prompt` and `Chatbot's History` and `Evaluation Details` very carefully.
            - If you think the `Chatbot's Current Instructions Prompt` is not optimized, when you consider the `Chatbot's History` and `Evaluation Details`, you can use `optimize_prompt` tool to optimize the prompt.
                    Example: In the `Chatbot's History`, when chatbot asks question, and user not responding directly, and asking for same question differenct format, make sure you can optimize the prompt to avoid asking same question again.
                      
                        1. When user ask content with different format, change the prompt accordingly.
                            Chatbot : This is your last 5 transactions. Select one of the following:
                                1. Adidas 2025-01-01 $500
                                2. Nike 2025-01-02 $600
                                3. Puma 2025-01-03 $700  
                                4. Reebok 2025-01-04 $800
                                5. Under Armour 2025-01-05 $900
                            User : No need to show all details. Just show me the merchant name only.
                            Chatbot : This is your last 5 transactions. Select one of the following:
                                1. Adidas
                                2. Nike
                                3. Puma
                                4. Reebok
                                5. Under Armour

                        2. When user ask additional information when chatbot asking question, change the prompt accordingly. Update prompt to avoid asking same question again next time.
                            Chatbot : This is your last 5 transactions. Select one of the following:
                                1. Adidas 2025-01-01 $500
                                2. Nike 2025-01-02 $600
                                3. Puma 2025-01-03 $700
                                4. Reebok 2025-01-04 $800
                                5. Under Armour 2025-01-05 $900

                            User : Please show me with transaction time also.
                            Chatbot : This is your last 5 transactions. Select one of the following:
                                1. Adidas 2025-01-01 11:58:00 $500
                                2. Nike 2025-01-02 12:00:00 $600
                                3. Puma 2025-01-03 12:01:00 $700
                                4. Reebok 2025-01-04 12:02:00 $800
                                5. Under Armour 2025-01-05 12:03:00 $900

                        3. If user request next time when he log in to the system, make sure to ask the same thing, change the prompt accordingly.
                            Chatbot : This is your last 5 transactions. Select one of the following:
                                1. Adidas 2025-01-01 $500
                                2. Nike 2025-01-02 $600
                                3. Puma 2025-01-03 $700  
                                4. Reebok 2025-01-04 $800
                                5. Under Armour 2025-01-05 $900
                            User : No need to show all details. Just show me the merchant name only. Next time when i log in to the system, make sure to ask the same thing.
                            Chatbot : This is your last 5 transactions. Select one of the following:
                                1. Adidas
                                2. Nike
                                3. Puma
                                4. Reebok
                                5. Under Armour



            - Make sure when you consider the `Chatbot's History`, If it's meets above given criteria and example, you can use `optimize_prompt` tool to optimize the prompt.
                            

    """ 
    
    # Define the task for the agent
    test_chatbot_task = Task(
        description=test_bot_prompt,
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
        tools=[],
        callback=lambda result: globals().update({'evaluation_details': result.raw})
    )


    optimization_task = Task(
        description=optimization_propmt,
        agent=testing_agent,  # Assign the testing agent to this task
        expected_output="""
        If optimization is needed, provide the optimized prompt.
        (When user request same thing again with different format)
            OPTIMIZED_PROMPT: <The optimized prompt>
        
        """,
        tools=[optimize_prompt]
    )
    
    # Create a crew with the testing agent and task
    crew = Crew(
        agents=[testing_agent],
        tasks=[test_chatbot_task, evaluate_task, optimization_task],
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
        st.write("### Test Evaluation Report")
        # result_text = str(result.raw)
        text1 = str(make_payment_prompt)
        text2 = str(result.raw)

        # Display evaluation details
        st.markdown("##### Evaluation Details")
        st.write(evaluation_details)



        st.markdown("##### Differences Between Prompts")

        with st.expander("Original Prompt"):
            st.code(text1, language="markdown")

        with st.expander("Updated Prompt"):
            st.code(text2, language="markdown")
        
        diff_html = get_diff_html(text1, text2)
        st.markdown(diff_html, unsafe_allow_html=True)

        
    
        # score_match = re.search(r"(Score:\s*.*?)(?=Criteria|$)", result_text, re.DOTALL)
        # criteria_match = re.search(r"(Criteria:\s*.*?)(?=Supporting Evidence|$)", result_text, re.DOTALL)
        # evidence_match = re.search(r"(Supporting Evidence:\s*.*)", result_text, re.DOTALL)

        # score_text = score_match.group(1).strip() if score_match else "Score: N/A"
        # criteria_text = criteria_match.group(1).strip() if criteria_match else "Criteria: N/A"
        # evidence_text = evidence_match.group(1).strip() if evidence_match else "Supporting Evidence: N/A"

        # st.write(score_text)
        # st.write(criteria_text)
        # st.write(evidence_text)
        # st.write(text2)

# Run the Streamlit app
if __name__ == "__main__":
    main()