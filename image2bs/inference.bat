@echo off
e:
cd E:\Projects\Pycharm-projects\facial_expression_parameter\image2bs
SET PYTHON_PATH=E:\Tools\Anaconda\Anaconda3\envs\py38
SET INF_PATH=E:\Projects\Pycharm-projects\facial_expression_parameter\image2bs
SET PYTHONEXECUTABLE=%PYTHON_PATH%\python.exe
"%PYTHONEXECUTABLE%" "%INF_PATH%"\inference.py
pause
