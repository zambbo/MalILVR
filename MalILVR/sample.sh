export PYTHONPATH=$PYTHONPATH:$(pwd)

SAVE_DIR=output
MODEL_PATH=models/ddpm_100.pt
SAMPLE_PATH=ref_imgs/benigns

python3 scripts/ilvr_sample.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 64 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 False --timestep_respacing 100 --model_path $MODEL_PATH --base_samples $SAMPLE_PATH --down_N 32 --range_t 20 --save_dir $SAVE_DIR
