@echo off
echo Starting Smart Maternal Health Companion...

echo Starting Backend Server...
start "Maternal Companion Backend" cmd /k "cd backend && python app.py"

echo Starting Frontend Server...
start "Maternal Companion Frontend" cmd /k "cd frontend && npm run dev"

echo Waiting for servers to initialize...
timeout /t 5

echo Opening Application...
start http://localhost:5173

echo.
echo.
echo Application started successfully!
echo Access the application at: http://localhost:5173
echo.
pause
