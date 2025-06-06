#!/bin/bash

# Activate conda environment
source activate adk-new-env

# Start Python HTTP server in background
python -m http.server 8090 &
SERVER_PID=$!

# Start ADK web server in background
adk web &
ADK_PID=$!

# Wait for 5 seconds
sleep 5

# Open the browser
open "http://localhost:8000/dev-ui?app=application"

# Keep the script running and handle cleanup on exit
trap "kill $SERVER_PID $ADK_PID" EXIT
wait 