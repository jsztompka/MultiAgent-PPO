
## Report

## Agent is using Proximal Policy Optimization Algorithm with Beta distribution. 

Proximal Policy Optimization:
PPO strikes a balance between ease of implementation, sample complexity, and ease of tuning, trying to compute an update at each step that minimizes the cost function while ensuring the deviation from the previous policy is relatively small.

Beta distribution is slight divergence from the original paper but in my experiments it made significant improvement. 

This algorithm uses function aproximator in a form of neural network. 

The first couple layers calculate policy distribution and return log(policy)
Actions are sampled from the beta distribution built using two layer outputs Alpha and Beta

Value is calculated from a separate head in the network Critic network part

The implemenation is based on the implementation from the paper but it uses Huber-Loss loss function to calculate the cost(loss). In my experiments Huber-Loss had better performance over standard loss functions.

On average my best score was between around 2.35 points

Config also includes all hyperparemeters which I found to work best.

## Model 

Model consists of several components, each built of 2 layers per component (body and head), body normally has 500 and head 400 nodes, details of the model are in the table below. 

|Layer (type)   |           Output Shape   |      Param #|
| --- | --- | --- | 
|       Inputs            |    [-1, 1000]         |	 25,000
|   BatchNorm1d-2         |        [-1, 1000]     |	      2,000
|        PolicyBody       |           [-1, 500]   |	      500,500
|   BatchNorm1d-4         |         [-1, 500]     |	      1,000
|        PolicyHead       |          [-1, 400]    |	     200,400
|        Actor            |      [-1, 400]        |	 160,400
|        Alpha            |        [-1, 2]        |	     802
|        Beta             |       [-1, 2]         |	    800
|        Actions          |          [-1, 2]      |	       800
|       Critic            |      [-1, 400]        |	 160,400
           
Total params: 1,052,503  
Trainable params: 1,052,503  
Non-trainable params: 0  

## Training chart: 
![](/images/Training.png)

## Parameters used (please see config.py): 
### Agent / network specific

 gae_tau = 0.95
 gradient_clip = 4.7
 rollout_length = 2000       - trajectory length when recording actions / states
 optimization_epochs = 10    - training length
 mini_batch_size = 500       - batch size in training
 ppo_ratio_clip = 0.123      - PPO clipping value 
 lr = 0.0001                 - initial learning rate decayed over time

## Future improvements: 
Better algorithm such as Soft Actor Critic could perform better. 
