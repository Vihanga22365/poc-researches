@echo off

:: Activate the conda environment
call conda activate agent-bot-env
timeout /t 0 >nul  # Wait for 3 seconds

:: Start the Single Intent Agent and keep the terminal open
start cmd /k "python main_agentic_chatbot.py"
timeout /t 1 >nul  # Wait for 3 seconds

:: Start the Test Agent and keep the terminal open
start cmd /k "streamlit run main_test_agent.py"
timeout /t 3 >nul  # Wait for 3 seconds

:: Open index.html in the default web browser
start index.html
timeout /t 3 >nul  # Wait for 3 seconds