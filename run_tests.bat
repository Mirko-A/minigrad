@echo off

cd tests
for %%i in (*.py) do (
    echo.
    if not "%%i"=="__init__.py" (
        echo Running tests for %%i
        python "%%i"
    )
)