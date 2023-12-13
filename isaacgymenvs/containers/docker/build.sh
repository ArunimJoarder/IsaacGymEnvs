#! /bin/bash
home=`realpath "$(dirname "$0")"/../../../../`
cd $home && sudo docker build --network=host --memory-swap -1 -m 20g -t deXtreme-thesis -f IsaacGymEnvs/isaacgymenvs/containers/docker/Dockerfile .
