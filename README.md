# CIFAR-Aware GreedyViG: Improving Vision GNNs with Adaptive Graph Construction

## Overview
This project extends the GreedyViG (CVPR 2024) architecture by adapting it for CIFAR-100 and introducing several architectural improvements.

We modify both the graph construction module and local feature extraction pipeline to improve performance on small-resolution images.

## Results

| Model | Top-1 Accuracy |
|------|---------------|
| Baseline GreedyViG | 51.7% |
| Modified GreedyViG (Ours) | 72.9% |

**+21.2% absolute improvement on CIFAR-100**

## Contributions

- CIFAR-specific adaptation (small-input stem + normalization)
- Learnable graph threshold with differentiable masking
- SE channel attention in local blocks
- Dual aggregation (max + mean)
- 5×5 depthwise convolution for improved receptive field

## Architectural Improvements

### 1. CIFAR Adaptation
- Modified stem to preserve spatial resolution
- Dataset-specific normalization

### 2. Graph Construction
- Learnable threshold (alpha)
- Soft sigmoid masking
- Fixed loop bug

### 3. Local Feature Learning
- SE attention
- Kernel size correction
- 5×5 depthwise convolution

### 4. Aggregation
- Combined max + mean aggregation

## Training

```bash
main.py \
  --data-set CIFAR \
  --input-size 32 \
  --model GreedyViG_S_CIFAR \
  --epochs 100 \
  --batch-size 128 \
  --distillation-type none \
  --warmup-epochs 5 \
  --aa rand-m7-mstd0.5-inc1 \
  --mixup 0.4 \
  --cutmix 0.5 \
  --reprob 0.1 \
  --model-ema-decay 0.9992 \
  --output_dir outputs
```



---

## Base Paper
```md
## Base Paper

GreedyViG: Dynamic Axial Graph Construction for Efficient Vision GNNs  
CVPR 2024

https://openaccess.thecvf.com/content/CVPR2024/html/Munir_GreedyViG_Dynamic_Axial_Graph_Construction_for_Efficient_Vision_GNNs_CVPR_2024_paper.html

## Acknowledgement

This project is based on the official GreedyViG implementation.
