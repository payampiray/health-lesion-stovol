import numpy as np


class ParticleFilter:
    def __init__(self, transition_function, observation_function, initial_state, resampling_method='systematic'):
        self.transition_function = transition_function
        self.observation_function = observation_function

        if resampling_method == 'systematic':
            self.resampling = self.resampling_systematic
        elif resampling_method == 'binomial':
            self.resampling = self.resampling_binomial
        else:
            raise ValueError('Unknown resampling method. It should be systematic or binomial')

        if initial_state.ndim != 2:
            raise ValueError('initial_state should be a num_dimensions x num_particles array')

        particles = initial_state
        self.num_dimensions = particles.shape[0]
        self.num_particles = particles.shape[1]
        self.particles = particles
        self.weights = np.ones(self.num_particles)/self.num_particles

    def predict(self, variables):
        particles = self.particles
        particles = self.transition_function(particles, variables)
        self.particles = particles
        state = particles@self.weights
        return state

    @staticmethod
    def resampling_systematic(w):
        num_particles = len(w)
        u = np.random.random()/num_particles
        edges = np.concatenate((0, np.cumsum(w)), axis=None)
        samples = np.arange(u, 1,step=(1/num_particles))
        idx = np.digitize(samples,edges)-1
        return idx

    @staticmethod
    def resampling_binomial(w):
        N = len(w)
        u = np.random.random(N)
        qc = np.cumsum(w)
        qc = qc/qc[-1]
        uqc = np.concatenate([u, qc])
        ind1 = np.argsort(uqc)
        ind2 = np.flatnonzero(ind1 < N)
        idx = ind2 - np.arange(N)
        return idx

    def correct(self, observation, variables):
        particles = self.particles

        likelihood = self.observation_function(particles, observation, variables)
        weights = self.weights
        weights = weights*likelihood
        weights = weights/sum(weights)

        N_eff = 1/np.sum(weights**2)
        resample_percent = 0.50
        Nt = resample_percent*self.num_particles
        if N_eff < Nt:
            idx = self.resampling_systematic(weights)
            weights = np.ones(self.num_particles)/self.num_particles
            particles = particles[:, idx]
        self.particles = particles
        self.weights = weights

    def filter(self, observations, variables):
        num_observations = len(observations)
        state_mean = np.zeros((num_observations, 1))
        weights = np.zeros((num_observations, self.num_particles))
        for t in range(num_observations):
            self.predict(variables)
            self.correct(observations[t], variables)
            state_mean[t,:] = self.particles@self.weights
            weights[t,:] = self.weights
        return state_mean, weights

