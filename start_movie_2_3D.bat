echo MDVT GUI

set "CONDA=%UserProfile%\miniconda3\condabin\conda.bat"
CALL "%UserProfile%\miniconda3\Scripts\activate.bat" mdvt


python MDVT_gui.py
pause