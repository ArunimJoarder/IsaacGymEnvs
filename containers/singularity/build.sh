#! /bin/bash
home=`realpath "$(dirname "$0")"`
cd $home && sudo singularity build --sandbox deXtreme.sif deXtreme.def
