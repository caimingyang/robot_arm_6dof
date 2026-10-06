@echo off
chcp 65001 >nul
echo ==========================================
echo  VLA闭环 - Qwen大模型+双目视觉
echo ==========================================
.venv\Scripts\python.exe run.py vla-qwen %*
pause
