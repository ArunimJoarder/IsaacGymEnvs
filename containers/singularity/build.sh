#! /bin/bash
home=`realpath "$(dirname "$0")"`
cd $home
sudo docker image save dextreme -o dextreme.tar && sudo singularity build --sandbox deXtreme.sif docker-archive:dextreme.tar &> build.log
