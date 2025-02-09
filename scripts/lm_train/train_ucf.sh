NGPUs=$1
NNodes=$2

# base model, 64 40G gpus in total, 256 bsz in total
NLayer=24
NHead=16
NEmbd=1536
MODEL='base'
BSZ=4

# # xl model, 64 80G gpus in total, 256 bsz in total
# NLayer=24
# NHead=32
# NEmbd=2048
# MODEL='xl'
# BSZ=4

# # 3b model, 64 80G gpus in total, 256 bsz in total
# NLayer=32
# NHead=32
# NEmbd=3072
# MODEL='3b'
# BSZ=4

LR=6e-4
BLOCK=16

# torchrun --nproc_per_node=$1 --nnodes=$2 \
#     --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT \
torchrun --nproc_per_node=$1 --nnodes=$2 --master_port="9778" \
    transformer_train.py --tokenizer "magvit2" --num_workers 32 --progress_bar_refresh_rate 500  \
    --num_nodes $2 --gpus $1 --sync_batchnorm --batch_size ${BSZ} --cond_stage_key 'text' --starts_with_bov \
    --base_lr ${LR} --lr_min 0 --warmup_steps 5000 --warmup_lr_init 0. \
    --tokenizer_path '/path/to/NBP/ckpt/NBP-tokenizer-ucf/magvit2_ucf.pt' --default_root_dir "/path/to/NBP/ckpt/ucf_semi_ar_${MODEL}_lr${LR}_steps100k_block${BLOCK}_hybrid" \
    --loader_type 'joint' --data_path '/path/to/NBP/data' --train_datalist '/path/to/NBP/annotations/ucf_train.txt' \
    --val_datalist '/path/to/NBP/annotations/ucf_val.txt' \
    --vocab_size 64000 --block_size 5120 --n_layer ${NLayer} --n_head ${NHead} --n_embd ${NEmbd} \
    --resolution 128 --sequence_length 17 --max_steps 100000 \
    --semi_ar --token_number_per_step ${BLOCK} \
    --llama_tokenizer "/path/to/NBP/llama_tokenizer/" --hybrid \
    --fps 8 --cond_token_num 8 \
    --best_checkpoint_monitor 'train/loss' \
    --bf16 --use_deepspeed # --grad_accumulates 2  # for 3b model
