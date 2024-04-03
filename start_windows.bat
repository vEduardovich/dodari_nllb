@echo off

if not exist "%~dp0\venv\Scripts" (
    echo ó�� ����� ���ٸ� ���� ȯ����� ������ �մϴ�.
    echo ���࿡ �ʿ��� ���ϵ��� ��ġ�մϴ�.. �ð��� �� �����ɸ��ϴ�.
    python -m venv venv
	cd /d "%~dp0\venv\Scripts"
    call activate.bat

    cd /d "%~dp0"
    pip install -r requirements.txt

    echo.
    echo ���ٸ� ����ȯ���� ����µ� �����߽��ϴ�!
    echo.
)

if errorlevel 1 (
    echo.
    echo ȯ�� ������ �����߽��ϴ�. ��Ȥ �̷����� �ֽ��ϴ�.
    echo "venv ������ �����ϰ� start_windows.bat�� �ٽ� �����غ�����.
    pause
)

echo ��뷮 ������ ���ٸ��� �����մϴ�.
echo ��ø� ��ٷ��ּ���..
goto :activate_venv

:launch
%PYTHON% dodari.py
pause

:activate_venv
set PYTHON="%~dp0\venv\Scripts\Python.exe"
goto :launch
:endofscript

echo.
echo ���࿡ �����߽��ϴ�. â�� �ݽ��ϴ�.
pause