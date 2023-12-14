#!/bin/bash

# This script can be used to debug the container locally
# It will mount the current directory into the container
# and run a shell inside the container. This allows you to
# run the container locally and debug it inside. (thanks Github Copilot)
DEXTREME_WS="${PWD}/../../../"

custom_flags="--nv --writable -B $DEXTREME_WS/IsaacGymEnvs:/DEXTREME_WS/IsaacGymEnvs"

echo $custom_flags

singularity shell $custom_flags $DEXTREME_WS/IsaacGymEnvs/isaacgymenvs/containers/singularity/deXtreme.sif
