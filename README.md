# FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding

## Abstract

Although Domain Adaptation in Semantic Scene Segmentation has shown impressive improvement in recent years, 
the fairness concerns in the domain adaptation have yet to be well defined and addressed. 
In addition, fairness is one of the most critical aspects when deploying the segmentation models into human-related real-world applications, 
e.g., autonomous driving, as any unfair predictions could influence human safety. 
In this paper, we propose a novel Fairness Domain Adaptation (FREDOM) approach to semantic scene segmentation. 
In particular, from the proposed formulated fairness objective, a new adaptation framework will be introduced based on the fair treatment of class distributions. 
Moreover, to generally model the context of structural dependency, a new conditional structural constraint is introduced to impose the consistency of predicted segmentation. 
Thanks to the proposed Conditional Structure Network, the self-attention mechanism has sufficiently modeled the structural information of segmentation. 
Through the ablation studies, the proposed method has shown the performance improvement of the segmentation models and promoted fairness in the model predictions. 
The experimental results on the two standard benchmarks, i.e., SYNTHIA to Cityscapes and GTA5 to Cityscapes, 
have shown that our method achieved State-of-the-Art (SOTA) performance.

## Installation and Usage

The detail of our source code will be available soon.


## Citation
If you find this code useful for your research, please consider citing:
```
@inproceedings{truong2023fredom,
  title={FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding},
  author={Truong, Thanh-Dat and Le, Ngan and Raj, Bhiksha and Cothren, Jackson and Luu, Khoa},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
