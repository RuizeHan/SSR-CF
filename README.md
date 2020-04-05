# SSR-CF

Public code of SSR-CF (Selective Spatial Regularization by Reinforcement Learned Decision Making for Object Tracking), published in TIP 2020.
Please see the project page (https://github.com/tsingqguo/ccotssr) for details.

```
@article{guo2020Selective,
  title={Selective Spatial Regularization by Reinforcement Learned Decision Making for Object Tracking}, 
  author={Guo, Qing and Han, Ruize and Feng, Wei and Chen, Zhihao and Wan, Liang},  
  year={2020},  
  journal={IEEE Transactions on Image Processing}
}
```

## Introduction

In this work, we propose selective spatial regularization (SSR) for CF-tracking scheme. It can achieve not only higher accuracy and robustness, but also
higher speed compared with spatially-regularized CF trackers. Specifically, rather than simply relying on foreground information, we extend the objective function of CF tracking scheme to learn the target-context-regularized filters using target-contextdriven weight maps. We then formulate the online selection of these weight maps as a decision making problem by a Markov Decision Process (MDP), where the learning of weight map selection is equivalent to policy learning of the MDP that is solved by
a reinforcement learning strategy. Moreover, by adding a special state, representing not-updating filters, in the MDP, we can learn when to skip unnecessary or erroneous filter updating, thus accelerating the online tracking. Finally, the proposed SSR is used to equip three popular spatially-regularized CF trackers to significantly boost their tracking accuracy, while achieving much
faster online tracking speed.


