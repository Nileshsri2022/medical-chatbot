@echo off
chcp 65001 >nul 2>&1
setlocal

echo ====================================================
echo   Medical RAG Chatbot - Local Startup (Windows)
echo ====================================================
echo.

set PROJECT_ROOT=%~dp0..
set BACKEND_DIR=%PROJECT_ROOT%\backend
set FRONTEND_DIR=%PROJECT_ROOT%\frontend

:: Check if .env exists
if not exist "%PROJECT_ROOT%\.env" (
    echo [ERROR] .env file not found!
    echo.
    echo Step 1: Copy .env.example to .env:
    echo    copy .env.example .env
    echo.
    echo Step 2: Edit .env and add your API key
    echo.
    echo Step 3: Run this script again
    echo.
    pause
    exit /b 1
)

echo [1/3] Starting LLM Backend on port 8001...
start "LLM Server" cmd /k "cd /d %BACKEND_DIR% && python llm_server.py"
timeout /t 4 /nobreak >nul

echo [2/3] Starting RAG Enhancement Server on port 8002...
start "RAG Server" cmd /k "cd /d %BACKEND_DIR% && python medical_rag_server.py"
timeout /t 4 /nobreak >nul

echo [3/3] Starting Frontend Server on port 3000...
start "Frontend Server" cmd /k "cd /d %FRONTEND_DIR% && python -m http.server 3000"
timeout /t 2 /nobreak >nul

echo.
echo ====================================================
echo   All servers started!
echo ====================================================
echo.
echo   Frontend:    http://localhost:3000/enhanced-medical-chatbot.html
echo   RAG Server:  http://localhost:8002  (docs: http://localhost:8002/docs)
echo   LLM Server:  http://localhost:8001  (docs: http://localhost:8001/docs)
echo.
echo   Close the 3 server windows to stop all services.
echo ====================================================
echo.
echo Opening browser...
start http://localhost:3000/enhanced-medical-chatbot.html
