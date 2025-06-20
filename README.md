# Action Curiosity-Based DRL for Path Planning

This repository contains the official implementation of our paper:

**"Action-Curiosity-Based Deep Reinforcement Learning Algorithm for Path Planning in a Nondeterministic Environment"**,  
*Junxiao Xue, Jinpu Chen, Shiwen Zhang*  
Published in *Intelligent Computing*, Vol. 4, 2025  
[DOI: 10.34133/icomputing.0140](https://doi.org/10.34133/icomputing.0140)

Our method introduces an action curiosity module and a cosine annealing strategy to improve exploration and stability in DRL-based path planning under dynamic and uncertain conditions.


# How to train
### 1.Environmental configuration

The code needs to run under Ubuntu20.04, installation tutorial may refer to: https://blog.csdn.net/Cui_Hongwei/article/details/109438310.

On this basis, the code needs to the construction of the environment, ROS robot installation tutorial may refer to: https://blog.csdn.net/qq_33361420/article/details/118222009.

The software packages required during the code execution can be installed according to the instructions in run.txt. 

The main process code is in ~/catkin_ws/src/turtlebot3_drl_pp/


### 2.Operation process

After the environment is set up, please type the following commands in sequence in the linux command line to run the code

cd ~/catkin_ws

catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.9

roslaunch turtlebot3_drl_pp stage_xx_xx.launch


# Citation
If you find this project useful in your research, please consider citing our paper:

```bibtex
@article{xue2025action,
  title={Action-Curiosity-Based Deep Reinforcement Learning Algorithm for Path Planning in Non-deterministic Environment},
  author={Xue, Junxiao and Chen, Jinpu and Zhang, Shiwen},
  journal={Intelligent Computing},
  year={2025},
  publisher={AAAS}
}
