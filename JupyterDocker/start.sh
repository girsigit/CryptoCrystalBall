CURRENTDIR=$PWD
PARENTDIR="$(dirname "$CURRENTDIR")"

echo "${CURRENTDIR}"
echo "${PARENTDIR}"

docker run \
-i \
-t \
--rm \
-v ${CURRENTDIR}/data/:/home/jovyan/work/data/ \
-v ${CURRENTDIR}/notebooks/:/home/jovyan/work/notebooks/ \
-v ${CURRENTDIR}/user-settings:/home/jovyan/.jupyter/lab/user-settings \
-v ${PARENTDIR}/:/home/jovyan/CryptoCrystalBall \
-p 8888:8888 \
--name jupdocker \
jupdocker-image
