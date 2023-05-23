export OPENAI_LOGDIR="./log"

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --num_head_channels 64 --learn_sigma True --class_cond False --attention_resolutions 16 --resblock_updown True --use_fp16 False"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 16 --save_interval 5000"
#CLASSIFIER_FLAGS="--image_size 64 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

python3 image_train.py --data_dir ../datasets/malwares/ $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
#mpiexec -n 1 python3 image_train.py --data_dir ../datasets/malwares/ $TRAIN_FLAGS #$CLASSIFIER_FLAGS
