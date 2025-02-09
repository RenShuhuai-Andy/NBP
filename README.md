# Next Block Prediction: Video Generation via Semi-Autoregressive Modeling

Official pytorch implementation of the following paper:
<p align="left"> 
<a href="https://arxiv.org/">Next Block Prediction: Video Generation via Semi-Autoregressive Modeling</a>.
<br>
<br>
<a href="https://renshuhuai-andy.github.io/">Shuhuai Ren</a><sup>1</sup>, <a href="https://www.microsoft.com/en-us/research/people/shumma/">Shuming Ma</a><sup>2</sup>, <a href="https://xusun26.github.io/">Xu Sun</a><sup>1</sup>, <a href="https://thegenerality.com/">Furu Wei</a><sup>2</sup>
<br>
<sup>1</sup>Peking University <br>
<sup>2</sup>Microsoft Research Asia
</p>

<p align="left">
    <img src=assets/framework.png width="852" height="300" />
</p>


We introduce a semi-autoregressive (semi-AR) framework, called **N**ext-**B**lock **P**rediction (**NBP**), for video generation. 
This framework features the following properties:

- 🚀 The generation unit is shifted from individual tokens to blocks (e.g., rows or frames), where each token in the current block simultaneously predicts the corresponding token in the next block;
- 🔥 By employing bidirectional attention within each block, enabling tokens to capture more robust spatial dependencies;
- ⚡ By predicting multiple tokens in parallel, NBP models significantly reduce the number of generation steps, leading to **11x** faster inference; 
- 🥇 **State-of-the-art generation performance** on video datasets compared to AR-based models;

Please refer to our [project page](https://RenShuhuai-Andy.github.io/NBP-project/) for the reconstruction and generation results by OmniTokenizer.

## Setup

### Enviroment
Please setup the environment using the following commands:

```
sh setup.sh
```

### Data & Model
Download the datasets from the official websites. You can download the [annotation.zip](https://huggingface.co/Daniel0724/OmniTokenizer/resolve/main/annotations.zip) processed by us and put them under ```./annotations```. 

The file structure should look like:

```
NBP/
    data/
        |–– UCF-101/
            |-- ApplyEyeMakeup/
                |-- v_ApplyEyeMakeup_g01_c01.avi
                |-- ...
            |-- ...
        |-- kinetics-dataset/
            |-- k600/
                |-- train/
                |-- val/
                |-- test/
    
    annotations/
    |-- ucf_train.txt
    |-- ucf_val.txt
    |-- k600_train.txt
    |-- k600_val.txt

    ckpt/
    |–– NBP-ucf-base/
        |-- ucf_base_nbp16_hybrid.ckpt
    |-- NBP-k600-base/
        |-- k600_base_nbp16.ckpt
    |–– NBP-tokenizer-ucf/
        |-- magvit2_ucf.pt
    |-- NBP-tokenizer-k600/
        |-- magvit2_k600.pt
```

## Model Zoo for Video Tokenizer
<p align="left">
    <img src=assets/tokenizer.png width="750" height="400" />
</p>
We reproduce closed-source MAGVITv2 as our video tokenizer. In contrast to the official implementation, which utilizes LFQ as its quantizer, we adopt FSQ due to its simplicity and reduced number of loss functions and hyper-parameters. Following the original paper's recommendations, we set the FSQ levels to $[8, 8, 8, 5, 5, 5]$, and the size of the visual vocabulary is 64K. 
Moreover, we employ PatchGAN instead of StyleGAN to enhance training stability.

 |  Training Data  | rFVD (128x128) | ckpt | 
 | ---------- | ----------- | ----------- | 
 | UCF-101 | 15.50 | [magvit2_ucf.pt](https://huggingface.co/ShuhuaiRen/NBP-tokenizer-ucf) |
 | K600 | 6.73 | [magvit2_k600.pt](https://huggingface.co/ShuhuaiRen/NBP-tokenizer-k600) | 

You can easily incorporate our tokenizer into your language model with:
```
from nbp.download import load_magvit2

# load tokenizer
tokenizer = load_magvit2('/path/to/tokenizer.pt', resolution=128, device="cuda")
tokenizer.eval()

# encode
_, tokens = tokenizer.encode(raw_video, quantize=True)
# decode
pred_video = tokenizer.decode_from_code_indices(tokens)
pred_video = torch.clamp(pred_video, 0, 1)
```

For the evaluation of our tokenizer, please refer to ```scripts/recons/eval_tokenizer.sh```.


## Model Zoo for Video Generator

|  Training Data  | Model Size | #Token | #step | gFVD (128x128) | ckpt | 
| ---------- | ----------- | ----------- | ----------- | ----------- | ----------- | 
| UCF-101 | 700M | 1280 | 95 | 103.3 | [ucf_base_nbp16_hybrid.ckpt](https://huggingface.co/ShuhuaiRen/NBP-ucf-base) |
| UCF-101 | 3B | 1280 | 95 | 55.3 | [ucf_3b_nbp16_hybrid.ckpt](https://huggingface.co/ShuhuaiRen/NBP-ucf-3b) |
| K600 | 700M | 768 | 48 | 25.5 | [k600_base_nbp16.ckpt](https://huggingface.co/ShuhuaiRen/NBP-k600-base) | 
| K600 | 3B | 768 | 48 | 19.5 | [k600_3b_nbp16.ckpt](https://huggingface.co/ShuhuaiRen/NBP-k600-3b) | 

### Training
Please refer to ```scripts/lm_train``` for model training. 

If you use deepspeed, after training, run
```
python zero_to_fp32.py /path/to/checkpoint-folder /path/to/output-file
```
to convert the model to fp32.

### Evaluation
Please refer to ```scripts/lm_eval``` for model evaluation. 

## Acknowledgments
Our code is partially built upon [OmniTokenizer](https://github.com/FoundationVision/OmniTokenizer) and
[FSQ-pytorch](https://github.com/duchenzhuang/FSQ-pytorch). 


## License

This project is licensed under the MIT license, as found in the LICENSE file.
