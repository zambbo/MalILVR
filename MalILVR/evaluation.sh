#!/bin/bash


SAMPLE_BASE_PATH=../MalGAN/generated_samples
BLACKBOX_BASE_PATH=../blackbox
LOG_BASE_PATH=../MalGAN/log_mal_gan

BLACKBOX_NAME_LIST=("DecisionTree" "LogisticRegression" "RandomForest" "SupportVectorMachine")

for blackbox_name in ${BLACKBOX_NAME_LIST[@]}
do
	sample_path=$SAMPLE_BASE_PATH/$blackbox_name
	blackbox_path=$BLACKBOX_BASE_PATH/${blackbox_name}.pickle
	log_path=$LOG_BASE_PATH/${blackbox_name}
	python3 evaluation.py --sample_path $sample_path --blackbox_path $blackbox_path --name $blackbox_name --log_path $log_path
done
