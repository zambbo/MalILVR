export OPENAI_LOGDIR="./log_guided"

TRAIN_FLAGS="--save_interval 500"

MODEL_FLAGS="--classifier_attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 64 --learn_sigma True --noise_schedule cosine"

python3 classifier_train.py --data_dir ../datasets/guided_train_dataset/ $MODEL_FLAGS $TRAIN_FLAGS
