#!/bin/bash

# exit if something fails!
# set -e

checkpoint=${HOME}/deXtreme/IsaacGymEnvs/isaacgymenvs/runs/dextreme_checkpoints/last_allegrohand_multigpuOVX_noFabrics_fixedFT_forceScale10_v5_ep_98000_rew_4717.542.pth

num_games=120000
num_envs=1024
timeout=1230

# properties=("adr.params.hand_damping" "adr.params.hand_stiffness" "adr.params.hand_joint_friction" "adr.params.hand_armature" "adr.params.hand_effort" "adr.params.hand_lower" "adr.params.hand_upper" "adr.params.hand_mass" "adr.params.hand_friction_fingertips" "adr.params.hand_restitution" "adr.params.object_mass" "adr.params.object_friction" "adr.params.object_restitution" "adr.params.cube_obs_delay_prob" "adr.params.cube_pose_refresh_rate" "adr.params.action_delay_prob" "adr.params.action_latency" "adr.params.affine_action_scaling" "adr.params.affine_action_additive" "adr.params.affine_action_white" "adr.params.affine_cube_pose_scaling" "adr.params.affine_cube_pose_additive" "adr.params.affine_cube_pose_white" "adr.params.affine_dof_pos_scaling" "adr.params.affine_dof_pos_additive" "adr.params.affine_dof_pos_white" "adr.params.rna_alpha")

# # min_value=("0.1140" "0.2715" "-0.3570" "-0.2085" "0.7155" "-4.6020" "-0.8010" "-0.3405" "-0.1350" "-0.1500" "0.0675" "0.0630" "-0.1500" "-0.1050" "0.7500" "-0.0885" "-0.2550" "0.0000" "-0.0555" "-0.0840" "0.0000" "-0.0285" "-0.0345" "0.0000" "-0.0420" "-0.0465" "-0.0285")
# # max_value=("4.5060" "2.5935" "3.9270" "2.4735" "4.3695" "0.9420" "4.4910" "3.9255" "3.2850" "1.6500" "2.4975" "3.2670" "1.6500" "1.1550" "9.7500" "0.9735" "2.8050" "0.0000" "0.6105" "0.9240" "0.0000" "0.3135" "0.3795" "0.0000" "0.4620" "0.5115" "0.3135")

# min_value=("-0.900" "-0.335" "-1.190" "-0.735" "-0.335" "-4.300" "-1.710" "-1.175" "-0.850" "-0.500" "-0.495" "-0.670" "-0.500" "-0.350" "-1.500" "-0.295" "-0.850" " 0.000" "-0.185" "-0.280" " 0.000" "-0.095" "-0.115" " 0.000" "-0.140" "-0.155" "-0.095" )
# max_value=("3.980" "2.245" "3.570" "2.245" "3.725" "1.860" "4.170" "3.565" "2.950" "1.500" "2.205" "2.890" "1.500" "1.050" "8.500" "0.885" "2.550" "0.000" "0.555" "0.840" "0.000" "0.285" "0.345" "0.000" "0.420" "0.465" "0.285" )

# properties=("forceScale" "forceScale" "forceScale" "forceScale" "forceScale" "forceScale" "forceDecay" "forceDecay" "forceDecay" "forceDecay" "forceDecay" "forceDecayInterval" "forceDecayInterval" "forceDecayInterval" "forceDecayInterval" "forceDecayInterval" "forceDecayInterval")
# value=("2.0" "2.2" "2.4" "2.6" "2.8" "3.0" "0.99" "0.79" "0.69" "0.59" "0.49" "0.080" "0.088" "0.096" "0.104" "0.112" "0.120")

properties=("forceProbRange" "forceProbRange" "forceProbRange" "forceProbRange" "forceProbRange")
min_value=("0.0009" "0.0008" "0.0007" "0.0006" "0.0005")
max_value=("0.1900" "0.2800" "0.3700" "0.4600" "0.5500")

main_dir=${HOME}/deXtreme/IsaacGymEnvs/isaacgymenvs
tmpfile=temp

property_names=A
filename=$tmpfile
re=0.5
le=$re
while getopts d:t:n:p:r:l:f:g: flag
do
    case "${flag}" in
        d) exp_dir=${OPTARG};;
        t) timeout=${OPTARG};;
        n) num_envs=${OPTARG};;
        p) property_names=${OPTARG};;
        r) re=${OPTARG};;
        l) le=${OPTARG};;
        f) filename=${OPTARG};;
        g) num_games=${OPTARG};;
    esac
done
# # echo $re $le $property_names $filename
# python $main_dir/scripts/generate_ranges.py -r $re -p $property_names -f $filename-$re

# filename=${main_dir}/scripts/ranges/$filename-$re.txt

# j=0
# while read F  ; do
#     line=$F
#     # echo $line
#     if [ $j -eq 0 ]
#     then
#         declare -a properties=( $line )
#     elif [ $j -eq 1 ]
#     then
#         declare -a min_value=( $line )
#     else
#         declare -a max_value=( $line )
#     fi
#     # echo $line
#     j=$((j + 1))
#     # echo $j
# done <$filename

len=${#properties[@]}
echo $properties $min_value $max_value $len
# temp_len=$max_len

# if [ $temp_len -gt $max_len ]
# then
#     echo "Number of properties tested should be less than " $max_len
#     exit 1
# else
#     len=$temp_len
# fi

# timeout $timeout \
#     python $main_dir/train.py \
#         task=AllegroHandDextremeADR \
#         checkpoint=${checkpoint} \
#         headless=True \
#         test=True \
#         num_envs=$num_envs \
#         task.env.printNumSuccesses=True \
#         task.experiment_dir=${exp_dir} \
#         task.experiment=base #\
#         # train.params.config.player.games_num=$num_games 

for(( i=0; i<$len; i++ ))
do
    echo "Sensitivity Analysis for noise in" ${properties[i]}
    timeout $timeout \
        python $main_dir/train.py \
            task=AllegroHandDextremeADR \
            checkpoint=${checkpoint} \
            headless=True \
            test=True \
            num_envs=$num_envs \
            task.env.printNumSuccesses=True \
            task.experiment_dir=${exp_dir} \
            task.experiment="${properties[i]} ${min_value[i]} ${max_value[i]}" \
            task.env.${properties[i]}=[${min_value[i]},${max_value[i]}] #\
            # task.experiment="${properties[i]} ${value[i]}" \
            # task.env.${properties[i]}=${value[i]} #\
            # task.experiment="${properties[i]} ${min_value[i]} ${max_value[i]}" \
            # task.task.adr.params.${properties[i]}.init_range=[${min_value[i]},${max_value[i]}] #\
            # train.params.config.player.games_num=$num_games 

done
