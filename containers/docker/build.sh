#! /bin/bash
home=`realpath "$(dirname "$0")"/../../../`
cd $home && sudo docker build --progress=plain --no-cache --network=host --memory-swap -1 -m 20g -t dextreme -f IsaacGymEnvs/containers/docker/Dockerfile . &> IsaacGymEnvs/containers/docker/build.log
