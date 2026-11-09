@echo off
setlocal
pushd "%~dp0" || exit /b 1

set "VENV=.venv-build\windows"
set "PYTHON=%VENV%\Scripts\python.exe"
set "WORK=build\windows"
set "DIST=dist\windows"

if not exist "%PYTHON%" (
    call :create_venv
    if errorlevel 1 goto :failed
)
"%PYTHON%" -m pip --disable-pip-version-check install -r requirements-build.txt
if errorlevel 1 goto :failed

if not exist "%WORK%" mkdir "%WORK%"
if errorlevel 1 goto :failed
set "PYINSTALLER_CONFIG_DIR=%CD%\%WORK%\cache"
"%PYTHON%" -m PyInstaller ^
    --name FilmClear --onedir --windowed --noconfirm --noupx ^
    --distpath "%DIST%" --workpath "%WORK%\work" --specpath "%WORK%" ^
    --collect-data cv2.data ^
    --icon "%CD%\assets\filmclear.ico" ^
    --add-data "%CD%\assets\filmclear.png:assets" ^
    --add-data "%CD%\models\face_detection_yunet_2022mar.onnx:models" ^
    --add-data "%CD%\models\face_detection_yunet_2023mar.onnx:models" ^
    --add-data "%CD%\models\YuNet-LICENSE.txt:models" ^
    main.py
if errorlevel 1 goto :failed

echo Built: "%CD%\%DIST%\FilmClear\FilmClear.exe"
popd
exit /b 0

:create_venv
if defined FILMCLEAR_PYTHON goto :custom_python
if exist ".venv\Scripts\python.exe" goto :project_python
where py >nul 2>nul
if errorlevel 1 goto :path_python
py -3 -m venv "%VENV%"
exit /b %errorlevel%

:custom_python
"%FILMCLEAR_PYTHON%" -m venv "%VENV%"
exit /b %errorlevel%

:project_python
".venv\Scripts\python.exe" -m venv "%VENV%"
exit /b %errorlevel%

:path_python
python -m venv "%VENV%"
exit /b %errorlevel%

:failed
set "BUILD_STATUS=%errorlevel%"
echo Build failed. See the output above. 1>&2
popd
exit /b %BUILD_STATUS%
