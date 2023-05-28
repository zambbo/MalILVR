#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

blackbox_sample_list=("benign_RandomForest" "benign_LogisticRegression" "benign_SupportVectorMachine" "benign_DecisionTree")

SAVE_DIR=generated_samples/aaa
MODEL_PATH=models/mal_ilvr.pt
SAMPLE_PATH=ref_imgs/benigns/
SAMPLE_NUM=6896


for str in "${blackbox_sample_list[@]}"
do
	echo "$str"

	SAVE_DIR=generated_samples/${str}_N2
	SAMPLE_PATH=datasets/${str}

	GENERATE_SCRIPT=$(cat << EOF
python3 scripts/ilvr_sample_new.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 False --timestep_respacing 100 --model_path $MODEL_PATH --base_samples $SAMPLE_PATH --down_N 2 --range_t 10 --save_dir $SAVE_DIR --num_samples $SAMPLE_NUM
EOF
)
	$GENERATE_SCRIPT

done
