@echo off
title WE.ED.IT - AI Beat-Sync Video Director
echo =============================================
echo        WE.ED.IT AI Director v1.0
echo  Music directs. AI cuts. You create.
echo =============================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
python -m pip install -r requirements.txt --quiet
python main.py
echo.
echo =============================================
echo Process finished. Videos saved in ./done
pause
