NET_NAME="supervised"
DATASET_NAME="deepl"
RESULT_DIR="/mnt/DATA/clma/semi"
EPOCHS=30
BATCH_SIZE=8
NET_DICT="dict(in_channels=1,out_channels=1,norm_type='INSTANCE')"

CUDA_VISIBLE_DEVICES="7" python -m torch.distributed.launch \
--master_port 18100 --nproc_per_node 1 main.py \
--lr 1e-4 --epochs $EPOCHS \
--milestones 20 40 50 --step_gamma 0.5 \
--net_name $NET_NAME --net_dict $NET_DICT \
--flip_prob 0.5 --rot_prob 0.5 \
--dataset_name $DATASET_NAME \
--num_train 0.9 --num_val 0.1 \
--min_hu -1024 --max_hu 3072 \
--loss_factor2 0.01 \
--qua_thres 7 --qua_thres2 10 \
--checkpoint_root "${RESULT_DIR}/ckpt" \
--tensorboard_root "${RESULT_DIR}/tb" \
--wandb_root "${RESULT_DIR}/wandb" \
--tensorboard_dir "${DATASET_NAME}_${NET_NAME}" \
--checkpoint_dir "${DATASET_NAME}_${NET_NAME}" \
--batch_size $BATCH_SIZE --num_workers 4 \
--log_interval 100 --save_epochs 1 \
--use_wandb --wandb_project 'risemar'




