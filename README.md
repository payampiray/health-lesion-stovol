# health-lesion-stovol
healthy and lesion models for learning based on the joint estimation of stochasticity and volatility

## author
Payam Piray (ppiray [at] princeton.edu)

## reference
please cite this paper if you use this code:
Piray P and Daw ND, 'A model for learning based on the joint estimation of stochasticity and volatility', 2021, Nature Communications.


## description of the models
This work addresses the problem of learning in noisy environments, in which the agent must draw inferences (e.g., about true reward rates) from observations (individual reward amounts) that are corrupted by two distinct sources of noise: process noise or volatility and observation noise or stochasticity. Volatility captures the speed by which the true value being estimated changes from trial to trial (modeled as Gaussian diffusion); stochasticity describes additional measurement noise in the observation of each outcome around its true value (modeled as Gaussian noise on each trial). The celebrated Kalman filter makes inference based on known value for both stochasticity and volatility, in which volatility and stochasticity have opposite effects on the learning rate (i.e. Kalman gain): whereas volatility increases the learning rate, stochasticity decreases the learning rate.

The learning models implemented here generalize the Kalman filter by also learning both stochasticity and volatility based on observations.
An important point is that inferences about volatility and stochasticity are mutually interdependent. But the details of the interdependence are themselves informative. From the learner’s perspective, a challenging problem is to distinguish volatility from stochasticity when both are unknown, because both of them increase the noisiness of observations. Disentangling their respective contributions requires trading off two opposing explanations for the pattern of observations, a process known in Bayesian probability theory as explaining away. This insight results in two lesion models: a stochasticity lesion model that tends to misidentify stochasticity as volatility and inappropriately increases learning rates; and a volatility lesion model that tends to misidentify volatility as stochasticity and inappropriately decreases learning rates.

## description of the code
learning_models.py contains two classes of learning models:
1) LearningModel that includes the healthy model and two lesion models (stochasticity lesion and volatility lesion models)
2) LearningModelGaussian is similar to LearningModel with the Gaussian generative processes for stochasticity and volatility diffusion.

Inference in both classes is based on a combination of particle filter and Kalman filter. Given particles for stochasticity and volatility, the Kalman filter updates its estimation of the mean and variance of the state (e.g. reward rate).
The main results shown in the reference paper (see below) is very similar for both classes of generative process. The particle filter has been implemented in the particle_filter.py

sim_example.py simulates the healthy model in a 2x2 factorial design (with two different true values for both true stochasticity and volatility). The model does not know about the true values and should learn them from observations. Initial values for both stochasticity and volatility are assumed to be the mean of their corresponding true values (and so not helpful for dissociation). This is akin to Figure 2 of the reference paper.

sim_lesion_example.py also simulates the lesions models in the 2x2 factorial design described above. This is akin to Figure 3 of the reference paper.


### Dependencies:
numpy (required for computations in particle_filter.py and learning_models.py)
matplotlib (required for visualization in sim_example and sim_lesion_example)
seaborn (required for visualization in sim_example and sim_lesion_example)
pandas (required for visualization in sim_example and sim_lesion_example)

### Other languages
The MATLAB implementation of this code (based on MATLAB's Control System Toolbox) is also available: https://github.com/payampiray/stochasticity_volatility_learning
