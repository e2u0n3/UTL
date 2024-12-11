## Universal Metric Learning

This is a code repository for universal metric learning. 

Note that the prompt learning code is not complete and is not uploaded now.

## Requirements

- Python3
- PyTorch (> 1.0)
- NumPy
- tqdm
- wandb
- timm
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)



## Run training, 1 dataset & 1 model without Prompt

```
python -m torch.distributed.launch --master_port=1234 --nproc_per_node=4 train.py \
                                                    --lr 1e-4 --batch_size 180 \
                                                    --warmup_epochs 1 --weight_decay 1e-4 --clip_grad 1.0 \
                                                    --loss PA --eval_freq 5  \
                                                    --IPC 0 --run_name PA \
                                                    --use_fp16 true --emb 128 --epochs 100 \
                                                    --model deit_small_distilled_patch16_224 --dataset All --freeze False
```


## Naive Prompt (only supports ViT)

```
python  train.py \
--lr 1e-4 --batch_size 180 \
--warmup_epochs 1 --weight_decay 1e-4 --clip_grad 1.0 \
--loss PA --eval_freq 5  \
--IPC 0 --run_name PA_NaivePrompt \
--use_fp16 true --emb 128 --epochs 100 \
--model vit_small_patch16_224 --dataset All --freeze True \
--prompt_type NaivePrompt --prompt_length 5

```

## Setup

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm wandb timm pytorch_metric_learning
```

## Datasets

- [CUB-200](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
- [Stanford Online Products](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
- [In-shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
    (In-shop Clothes Retrieval Benchmark -> Img -> img.zip, Eval/list_eval_partition.txt)
- [Cars-196](http://ai.stanford.edu/~jkrause/car196/car_ims.tgz) [labels](http://ai.stanford.edu/~jkrause/car196/cars_annos.mat)
