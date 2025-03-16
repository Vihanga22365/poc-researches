import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(
    page_title="CreditCard Recommendation Engine",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="collapsed"
    
)

st.subheader('CreditCard Recommendation Engine')

llm = ChatOpenAI(model_name = "gpt-4o",temperature=0.2)

#Pdf Upload Functionality
# uploaded_file = st.file_uploader("", type=['pdf'])
# st.session_state.pdf_context = ""
# if uploaded_file is not None:
#     with pdfplumber.open(uploaded_file) as pdf:
#         total_pages = len(pdf.pages)
#         full_text = ""
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 full_text += text + "\n" 
#         st.session_state.pdf_context = full_text

template = """You are an banking Sales Representative who will have conversation with potential customers to understand their spending habits and recommend bank’s credit cards available in a convincing manner.
Following context has information on available credit card products with their features. Answer the users question based on the product information.
Be proactive and ask questions from the user to understand life style, spending habits, gather information and see which product best fits the user and answer the questions very convincingly.
Limit your answer to 60 words or less at a time.
Your Name is Alex. introduce yourself as a Citi Bank’s Intelligent Agent and ask the customer's name at the first interaction. After customer’s reply, ask questions from the customer to understand passions, spending habits and gather information.
If customer is not convinced, elaborate the advantages of the products and how they outweigh the disadvantages and convince the customer to apply for the card. Do not offer sales features unless user ask for the offer.
Only reply as the sales representative and do not write the responses from the customer.
Answer only based on the topic of credit cards and If the customer questions is outside the context, just say that you don't know and steer the conversation back to the topic you know. Don't give any answer outside the context of credit cards.


Context: {pdf_context}

Current conversation:
{history}
Human: {input}
AI Assistant:"""

def generate_the_response(prompt, memory, pdf_context):
    PROMPT =  PromptTemplate.from_template(template).partial(pdf_context=pdf_context)
    llm_chain = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=memory,
    )
    result = llm_chain.predict(input=prompt)
    return result

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    
    #neuro-linguistic-recommendation-engine-healthcare-plans-chatbot {
        font-size: 22px;
        text-align: center;
    } 
    [data-testid="stChatInputTextArea"] {
        color: black;
        background: #ffffff;
        font-size: 20px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

main_context = """Following is a line of credit card products and their features.

Product 1: CITI AADVANTAGE CARD
Product 1 Features: Credit card offers 50,000 American Airlines AAdvantage bonus miles after $2,500 in purchases within the first 3 months of account opening. Earn 2 AAdvantage miles for every $1 spent at restaurants, gas stations, and eligible American Airlines purchases. Earn 1 AAdvantage mile for every $1 spent on all other purchases. Earn 1 Loyalty Point for every 1 eligible mile earned from purchases. First checked bag free on American Airlines domestic itineraries. Preferred boarding on American Airlines flights. 25% savings when you use your card on American Airlines inflight food and beverage purchases. $0 Intro Annual Fee, $99 after the first year. No foreign transaction fees.
Product 1 Sales Features: Sales person can offer one year waive off for annual fee.

Product 2: CITI CUSTOM CASH CARD
Product 2 Features: Earn 5% cash back on your top eligible spend category each billing cycle up to $500 spent. Earn 1% cash back thereafter on all other purchases. Special Travel Offer: Earn an additional 4% cash back on hotels, car rentals, and attractions booked on Citi Travel portal through 6/30/2025. 0% Intro APR for 15 months on purchases and balance transfers; after that, the variable APR will be 19.24% - 29.24%, based on your creditworthiness. Balance transfer fee of either $5 or 5% of the amount of each credit card balance transfer, whichever is greater. Balance Transfers must be completed within 4 months of account opening. No annual fee. $0 liability on unauthorized charges.

Product 3: CITI DIAMOND PREFERRED CREDIT CARD
Product 3 Features: 0% Intro APR for 21 months on balance transfers from date of first transfer; after that, the variable APR will be 18.24% - 28.99% based on your creditworthiness. Balance transfer fee of either $5 or 5% of the amount of each transfer, whichever is greater. 0% Intro APR for 12 months on purchases from date of account opening; after that, the variable APR will be 18.24% - 28.99% based on your creditworthiness. No annual fee. $0 liability on unauthorized charges.

Product 4: CITI DOUBLE CASH CARD
Product 4 Features: 0% Intro APR for 18 months on balance transfers; after that, the variable APR will be 19.24% - 29.24% based on your creditworthiness. Unlimited 1% cash back when you buy, plus an additional 1% as you pay, on every purchase. Earn $200 cash back after you spend $1,500 on purchases in the first 6 months of account opening. For a limited time, earn 5% total cash back on hotel, car rental and attractions when booked on the Citi TravelSM portal through 12/31/24. This is 3% cash back on top of the 1% when you buy and the 1% as you pay. No annual fee. $0 liability on unauthorized charges. 
Product 4 Sales Features: Sales person can offer late fee waive for first 3 months

Product 5: COSTCO ANYWHERE VISA CARD BY CITI
Product 5 Features: Earn 4% on eligible gas and EV charging with the Costco Anywhere Visa card for the first $7,000 per year and then 1% thereafter. Unlimited 3% on restaurants and eligible travel, including Costco Travel. Unlimited 2% on all other purchases from Costco and Costco.com. Unlimited 1% on all other purchases. Use the Costco Anywhere Visa card as your Costco membership ID. No annual fee with your active/paid Costco membership. No foreign transaction fees.
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages_memory = ConversationBufferMemory(memory_key="history")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can i help you?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages_memory = st.session_state.messages_memory
    # pdf_context = st.session_state.pdf_context
    pdf_context = main_context
    response = generate_the_response(prompt, messages_memory, pdf_context)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})