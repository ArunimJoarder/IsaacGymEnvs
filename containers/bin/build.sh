#!/bin/bash

# same as build_locally.sh. First, you need to extract the singularity image
# into your local scratch space. Then, you can debug on the cluster.
custom_flags="--nv --writable -B /cluster/home/$USER/git/IsaacGymEnvs/:/deXtreme/IsaacGymEnvs"

singularity shell $custom_flags $SCRATCH/deXtreme.sif