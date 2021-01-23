# How to run this project (on MacOS)

You need to have the docker daemon running so that docker can spin up a container.  
The way I do this on Mac is with docker-machine using the virtualbox driver.

## Install virtualbox, docker, and docker-machine

```shell
brew install docker
brew install --cask virtualbox
brew install docker-machine
```

the `--cask` flag is for installing brew formulae with GUIs.

## Start the docker daemon using docker machine

```shell
docker-machine create --driver virtualbox default
docker-machine start default
eval "$(docker-machine env default)"
```

If the `eval` command throws a `TSI connection` error, try running

```shell
docker-machine regenerate-certs default
```

and then try again. Note that the eval command makes the docker machine usable in the current shell tab. If you open up another terminal, make sure to run that eval command again in the new terminal.

## Build and run this project using docker

```shell
mkdir -p output

docker build -t cse447-proj/demo -f Dockerfile .

docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse447-proj/demo bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
```

## Stop the docker daemon when you're done

```shell
docker-machine stop default
```

## If you want to delete the docker machine when you're done

Delete the directory `/Users/<your_username>/.docker/machine/machines/default`.
