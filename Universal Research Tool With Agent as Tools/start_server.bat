@echo off
call conda activate adk-new-env
start cmd /k "python -m http.server 8090"
start cmd /k "adk web"
timeout /t 5
start http://localhost:8000/dev-ui?app=application 