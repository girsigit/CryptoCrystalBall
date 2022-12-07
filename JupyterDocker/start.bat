@echo off&setlocal
SET CURRENTDIR=%cd%

:: Get the parent dir for mapping folders
::https://stackoverflow.com/a/60046276
for %%I in ("%~dp0.") do for %%J in ("%%~dpI.") do set PARENTDIR=%%~dpnxJ

echo %CURRENTDIR%
echo %PARENTDIR%

@echo on

docker run ^
-i ^
-t ^
--rm ^
-v %CURRENTDIR%\data\:/home/jovyan/work/data/ ^
-v %CURRENTDIR%\notebooks\:/home/jovyan/work/notebooks/ ^
-v %CURRENTDIR%\user-settings:/home/jovyan/.jupyter/lab/user-settings ^
-v %PARENTDIR%\:/home/jovyan/CryptoCrystalBall ^
-p 8888:8888 ^
jupdocker
