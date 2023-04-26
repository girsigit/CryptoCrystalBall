CURRENTDIR=$PWD
PARENTDIR="$(dirname "$CURRENTDIR")"

echo "${CURRENTDIR}"
echo "${PARENTDIR}"
#echo "Password for the lab is 'crypto'"

docker run \
-i \
-t \
--rm \
-v ${PARENTDIR}/Data/:/content/dataset/ \
-v ${PARENTDIR}/Notebooks/:/content/notebooks/ \
-v ${CURRENTDIR}/user-settings:/home/jovyan/.jupyter/ \
-v ${PARENTDIR}/:/content/CryptoCrystalBall \
-v ${CURRENTDIR}/bigdata/:/content/bigdata/ \
-p 8888:8888 \
-p 6006:6006 \
--gpus all \
--name jupdocker \
jupdocker-image
