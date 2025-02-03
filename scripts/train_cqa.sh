NET_NAME="cqa"
DATASET_NAME="cqa"
RESULT_DIR="/mnt/DATA/clma/semi"
EPOCHS=30
BATCH_SIZE=1
NET_DICT="dict(use_rope=True,do_multiscale=True,in_channels=1,out_channels=10,attn_ratio=[0,1/2,1,0,0],drop_path_rates=0.1,use_spectrals=[True,True,True,False,False])"


CUDA_VISIBLE_DEVICES="7" python -m torch.distributed.launch \
--master_port 18888 --nproc_per_node 1 main.py \
--lr 1e-4 --epochs $EPOCHS \
--milestones 20 40 50 --step_gamma 0.5 \
--net_name $NET_NAME --net_dict $NET_DICT \
--flip_prob 0.5 --rot_prob 0.5 \
--dataset_name $DATASET_NAME \
--num_train 0.9 --num_val 0.1 \
--min_hu -1024 --max_hu 3072 \
--checkpoint_root "${RESULT_DIR}/ckpt" \
--tensorboard_root "${RESULT_DIR}/tb" \
--wandb_root "${RESULT_DIR}/wandb" \
--tensorboard_dir "${DATASET_NAME}_${NET_NAME}" \
--checkpoint_dir "${DATASET_NAME}_${NET_NAME}" \
--batch_size $BATCH_SIZE --num_workers 4 \
--log_interval 100 --save_epochs 1 \
--use_wandb --wandb_project 'cqa'

