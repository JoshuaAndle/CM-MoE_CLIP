# MoE-CLIP with Merging Experts for Online Continual Learning

## Introduction
Ongoing repository for work adapting MoE-CLIP to an online CL setting and implementing clustering and merging of experts for constrained model growth. This repository contains two directories, one for a prior, finished CNN implementation, and the second for a work-in-progress transformer implementation.


# CLIP Implementation
This work is currently being revised to work with CLIP in a multimodal transformer setting. The transformer code is based off of the code for "CLIP Model is an Efficient Online Continual Learner" located at https://github.com/Debatrix/LifeLong-CLIP. The original code provides the framework for OCL with a variety of benchmark methods. We extend this framework to an online continual-learning framework with distribution shifts between tasks (e.g. rotations, blur) and by implementing our own "CM-MoE" method. CM-MoE implements a variant of the clustering and merging algorithm used in the CNN implementation to consolidate similar or redundant experts and provides a bounded growth of the overall model. 



# CNN Implementation
An implementation of the underlying merging and clustering algorithm was initially made for use with CNNs, which has been included in the CNN directory. This method utilizes a mix of sample replay and structured pruning to improve efficiency of Online Continual Learning while retaining accuracy. Implementation improved accuracy compared to pruning along, while the training time needed was signficantly lower than an equivalent replay approach applied to the full model without structured pruning. Notably there are periodic spikes in runtime during merge operations due to the need for offline finetuning of merged subnetworks on replay buffer data. Once implementation on CLIP has finished we may revisit this setting to improve these results and provide comparison to benchmark methods.






@article{wang2024clip,
  title={CLIP model is an Efficient Online Lifelong Learner},
  author={Wang, Leyuan and Xiang, Liuyu and Wei, Yujie and Wang, Yunlong and He, Zhaofeng},
  journal={arXiv preprint arXiv:2405.15155},
  year={2024}
}
