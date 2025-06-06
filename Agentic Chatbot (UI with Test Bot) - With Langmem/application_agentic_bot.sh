#!/bin/bash

# Activate the conda environment
source activate helpshift-env
sleep 0  # Wait for 0 seconds (equivalent to your Windows timeout)

# Start the Single Intent Agent and keep the terminal open
osascript -e 'tell app "Terminal" to do script "python main_agentic_chatbot.py"'
sleep 1  # Wait for 1 second

# Start the Test Agent and keep the terminal open
osascript -e 'tell app "Terminal" to do script "streamlit run main_test_agent.py"'
sleep 3  # Wait for 3 seconds

# Open index.html in the default web browser
open index.html
sleep 3  # Wait for 3 seconds