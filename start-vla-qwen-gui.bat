@echo off
chcp 65001 >nul
echo ==========================================
echo  VLA闭环 - Qwen + MuJoCo可视化窗口
echo ==========================================
.venv\Scripts\python.exe run.py vla-qwen-gui %*
pause
