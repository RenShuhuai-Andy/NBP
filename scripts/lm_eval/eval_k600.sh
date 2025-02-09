nsample=50000

python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 transformer_eval.py --inference_type "video" \
                      --gpt_ckpt "/path/to/NBP/ckpt/NBP-k600-base/k600_base_nbp16.ckpt" \
                      --tokenizer_ckpt "/path/to/NBP/ckpt/NBP-tokenizer-k600/magvit2_k600.pt" \
                      --batch_size 1 --save "/path/to/NBP/save/k600_base_nbp16/eval${nsample}" --n_sample ${nsample} \
                      --top_k 16000 --top_p 0.9 --data_dir '/path/to/NBP/data/kinetics-dataset' --data_list '/path/to/NBP/annotations/k600_val.txt' \
                     --semi_ar --token_number_per_step 16 --tokenizer "magvit2" --resolution 128
