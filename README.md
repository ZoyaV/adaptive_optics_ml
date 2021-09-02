# adaptive_optics_ml
Adaptive optics machine learning experiments

In this repository, various experiments are presented to solve the problem of adaptive optics.

In the folder "architecture" in the Jupiter notebooks, various approaches to wavefront prediction are presented.

In total, four approaches are considered:

RESNET-50 - predicts the coefficients of the phase of aberrations;

UNET - predicts the total phase of aberrations;

CONV-LSTM - predicts the next frame or abbreviation from a series of previous frames; 

RL - Various approaches based on machine learning with reinforcement are considered (DDPG, PPO). Also on the link you can find a special environment for agent training (https://github.com/ZoyaV/ao_env).

# History and example of algorithms job

For each experiment there is a corresponding link to the project in WANDB

UNET-panel

![Section-1-Panel-0-2v2vj9etj](https://user-images.githubusercontent.com/10494404/131828536-52fa639b-8811-431d-bb12-be117af9ab7c.png)

In addition, each experiment is logged and the results are saved to the "history" folder.

![](/history/mse_conv_lstm_17_6_16_3.png)
