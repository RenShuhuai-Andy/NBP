NGPUs=$1
NNodes=$2

# base model, 32 40G gpus in total, 64 bsz in total
NLayer=24
NHead=16
NEmbd=1536
MODEL='base'
LR=1e-3
BSZ=2

# # xl model, 64 40G gpus in total, 64 bsz in total
# NLayer=24
# NHead=32
# NEmbd=2048
# MODEL='xl'
# LR=5e-4
# BSZ=1

# # 3b model, 32 80G gpus in total, 64 bsz in total
# NLayer=32
# NHead=32
# NEmbd=3072
# MODEL='3b'
# LR=2e-4
# BSZ=1

BLOCK=16

# torchrun --nproc_per_node=$1 --nnodes=$2 \
#     --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT \
torchrun --nproc_per_node=$1 --nnodes=$2 --master_port="9778" \
    transformer_train.py --tokenizer "magvit2" --num_workers 32 --progress_bar_refresh_rate 500  \
    --num_nodes $2 --gpus $1 --sync_batchnorm --batch_size ${BSZ} --unconditional \
    --base_lr ${LR} --lr_min 0 --warmup_steps 10000 --warmup_lr_init 0. \
    --tokenizer_path '/path/to/NBP/ckpt/NBP-tokenizer-k600/magvit2_k600.pt' --default_root_dir "/path/to/NBP/ckpt/k600_semi_ar_${MODEL}_lr${LR}_steps4m_block${BLOCK}" \
    --loader_type 'joint' --data_path '/path/to/NBP/data/kinetics-dataset' --train_datalist '/path/to/NBP/annotations/k600_train.txt' \
    --val_datalist '/path/to/NBP/annotations/k600_val.txt' \
    --vocab_size 64000 --block_size 5120 --n_layer ${NLayer} --n_head ${NHead} --n_embd ${NEmbd} \
    --resolution 128 --sequence_length 17 --max_steps 4000000 --semi_ar --token_number_per_step ${BLOCK} \
    # --bf16 --use_deepspeed --grad_accumulates 2  # for 3b model
