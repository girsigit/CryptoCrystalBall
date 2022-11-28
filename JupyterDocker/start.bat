SET CURRENTDIR="%cd%"
SET PARENTDIR="C:\Users\bgirsule\Documents\git\CryptoCrystalBall"

docker run -i -t --rm -v %CURRENTDIR%\data\:/home/jovyan/work/data/ -v %CURRENTDIR%\notebooks\:/home/jovyan/work/notebooks/ -v %CURRENTDIR%\user-settings:/home/jovyan/.jupyter/lab/user-settings -v %PARENTDIR%\:/home/jovyan/CryptoCrystalBall -p 8888:8888 jupdocker
