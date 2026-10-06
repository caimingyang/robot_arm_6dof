@echo off
chcp 65001 >nul
echo ==========================================
echo  VLA闭环 - Mock模式（无需API）
echo ==========================================
.venv\Scripts\python.exe run.py vla-mock %*
pause
