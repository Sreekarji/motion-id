@echo off
echo Starting Motion ID Demo...
echo.
echo Starting backend...
start "Motion ID Backend" cmd /k "cd D:\motionid\backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
timeout /t 4 /nobreak
echo.
echo Opening browser...
start http://localhost:8000
echo.
echo Demo running at http://localhost:8000
echo.
echo For external access (ngrok), open a new terminal and run:
echo   python -c "from pyngrok import ngrok; t=ngrok.connect(8000,'http'); print(t.public_url); input('Enter to stop')"
echo.
pause
