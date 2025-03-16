#!/bin/bash

# Activate the conda environment
source activate agent-bot-env
sleep 3  # Wait for 3 seconds

# Start the Single Intent Agent and keep the terminal open
gnome-terminal -- bash -c "python main_agentic_chatbot.py; exec bash"
sleep 1  # Wait for 1 second

# Start the Test Agent and keep the terminal open
gnome-terminal -- bash -c "streamlit run main_test_agent.py; exec bash"
sleep 3  # Wait for 3 seconds

# Open index.html in the default web browser
xdg-open index.html
sleep 3  # Wait for 3 seconds