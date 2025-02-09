cfg=0.0
nsample=10000
topk=16000
topp=0.9

python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1  transformer_eval.py --inference_type "video" \
                      --gpt_ckpt "/path/to/NBP/ckpt/NBP-ucf-base/ucf_base_nbp16_hybrid.ckpt" \
                      --tokenizer_ckpt "/path/to/NBP/ckpt/NBP-tokenizer-ucf/magvit2_ucf.pt" \
                      --batch_size 1 --save "/path/to/NBP/save/ucf_base_nbp16_hybrid/classcond_eval${nsample}" --n_sample ${nsample} --class_cond \
                      --top_k ${topk} --top_p ${topp} --data_dir '/path/to/NBP/data/' --data_list '/path/to/NBP/annotations/ucf_train.txt' \
                      --semi_ar --token_number_per_step 16 --tokenizer "magvit2" --resolution 128 \
                      --fps 8 --frm_sampling_strategy 'center' --hybrid \
                      --llama_tokenizer "/path/to/NBP/llama_tokenizer/"