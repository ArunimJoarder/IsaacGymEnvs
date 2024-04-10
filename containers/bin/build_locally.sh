#!/bin/bash

# This script can be used to debug the container locally
# It will mount the current directory into the container
# and run a shell inside the container. This allows you to
# run the container locally and debug it inside. (thanks Github Copilot)
DEXTREME_WS="`realpath "$(dirname "$0")"/../../../`"

custom_flags="--nv --writable -B $DEXTREME_WS/IsaacGymEnvs:/deXtreme/IsaacGymEnvs,$DEXTREME_WS/rl_games:/deXtreme/rl_games"

echo $custom_flags

singularity shell $custom_flags $DEXTREME_WS/IsaacGymEnvs/containers/singularity/deXtreme.sif
