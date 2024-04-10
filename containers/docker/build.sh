#! /bin/bash
home="/home/arunimj/docker_image_template"
cd $home && sudo docker build --progress=plain --no-cache --network=host --memory-swap -1 -m 20g -t dextreme -f /home/arunimj/deXtreme/IsaacGymEnvs/containers/docker/Dockerfile . &> /home/arunimj/deXtreme/IsaacGymEnvs/containers/docker/build.log
