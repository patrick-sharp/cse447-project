# Tests the model in a docker container. Before running this file, read the file `How to run docker.md`
docker-machine start default
eval "$(docker-machine env default)"
mkdir -p output
docker build -t cse447-proj/test -f Dockerfile .
docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse447-proj/test bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
docker-machine stop default