@echo off

set "PYTHONPATH=.."
set "PYTHONPATH=%PYTHONPATH%;../minitorch"
cd tests
echo Running tests:
python -m unittest discover