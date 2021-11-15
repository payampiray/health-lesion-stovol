import numpy as np
import particle_filter


class LearningModel:
    def __init__(self, lambda_stc, lambda_vol, init_stc=1.0, init_vol=1.0, x0_unc=100, num_particles=100,
                 resampling_method='systematic'):
        # lambda_stc and lambda_vol are update rate parameters for stochasticity and volatility, respectively.
        # These parameters should be in the unit range (although it is possible to set one of them as 0)
        # In practice, it is better to set these parameters in the range [0 0.25]
        # If either of these parameters are 0, this class creates the corresponding lesion model.
        # init_stc and init_vol are initial values (i.e. 0th trial) for stochasticity and volatility, respectively.
        # They both should be positive.
        # x0_unc is the initial value for Kalman variance. It should be positive.
        # num_particles determines the number of particles for the particle filter.
        # resampling determines the resampling method for the particle filter: systematic or binomial
        # The main method of this class is run(observations), which runs the constructed model for a timeseries observations
        # For an example usage of this class, see sim_example.py and sim_lesion_example.py

        self.lambda_stc = lambda_stc
        self.lambda_vol = lambda_vol
        self.init_stc = init_stc
        self.init_vol = init_vol
        self.x0_unc = x0_unc
        self.num_particles = num_particles
        self.resampling = resampling_method

        if lambda_stc == 0 and lambda_vol == 0:
            raise ValueError('Both lambda parameters are 0!')

        # if lambda_stc and lambda_vol are nonzero, this creates a healthy model
        # if lambda_stc is zero, this creates a stochasticity lesion model
        # if lambda_vol is zero, this creates a volatility lesion model
        if lambda_stc != 0 and lambda_vol != 0:
            self.model_name = 'healthy'
            self.model_type = [True, True]
            self.transition_sample = self._transition_sample
            self.observation_likelihood = self._observation_likelihood
            self.kalman_update = self._kalman_update
            initial_state = np.zeros((2, 1))
            initial_state[0] = 1/self.init_stc
            initial_state[1] = 1/self.init_vol
            self.initial_state = initial_state
        elif lambda_stc == 0:
            self.model_name = 'stochasticity lesion'
            self.model_type = [False, True]
            self.transition_sample = self._transition_sample_stc_lesion
            self.observation_likelihood = self._observation_likelihood_stc_lesion
            self.kalman_update = self._kalman_update_stc_lesion
            initial_state = np.zeros((1, 1))
            initial_state[0] = 1/self.init_vol
            self.initial_state = initial_state
        elif lambda_vol == 0:
            self.model_name = 'volatility lesion'
            self.model_type = [True, False]
            self.transition_sample = self._transition_sample_vol_lesion
            self.observation_likelihood = self._observation_likelihood_vol_lesion
            self.kalman_update = self._kalman_update_vol_lesion
            initial_state = np.zeros((1, 1))
            initial_state[0] = 1/self.init_stc
            self.initial_state = initial_state

        print('Model is %s' % (self.model_name,))

    @staticmethod
    def _transition_sample(particles, parameters):
        num_particles = particles.shape[1]

        y = particles[0, :]
        lambda_s = parameters[0]
        eta = 1-lambda_s
        nu = 0.5/(1-eta)
        epsil = (1/eta)*np.random.beta(eta*nu, (1-eta)*nu, num_particles)
        y = y*epsil

        z = particles[1, :]
        lambda_v = parameters[1]
        eta = 1-lambda_v
        nu = 0.5/(1-eta)
        epsil = (1/eta)*np.random.beta(eta*nu, (1-eta)*nu, num_particles)
        z = z*epsil

        particles[0, :] = y
        particles[1, :] = z
        return particles

    @staticmethod
    def _transition_sample_stc_lesion(particles, parameters):
        num_particles = particles.shape[1]
        z = particles[0, :]
        lambda_v = parameters[1]
        eta = 1-lambda_v
        nu = .5/(1-eta)
        epsil = (1/eta)*np.random.beta(eta*nu, (1-eta)*nu, num_particles)
        z = z*epsil

        particles[0, :] = z
        return particles

    @staticmethod
    def _transition_sample_vol_lesion(particles, parameters):
        num_particles = particles.shape[1]
        y = particles[0, :]
        lambda_s = parameters[0]
        eta = 1-lambda_s
        nu = .5/(1-eta)
        epsil = (1/eta)*np.random.beta(eta*nu, (1-eta)*nu, num_particles)
        y = y*epsil

        particles[0, :] = y
        return particles

    @staticmethod
    def _observation_likelihood(particles, observation, args):
        kalman_mean = args[0]
        kalman_var = args[1]
        s = 1/particles[0, :]
        v = 1/particles[1, :]
        var = kalman_var + v + s
        likelihood = np.exp(-1 / (2 * var) * ((observation - kalman_mean) ** 2)) / np.sqrt(2 * np.pi * var)
        return likelihood

    @staticmethod
    def _observation_likelihood_stc_lesion(particles, observation, args):
        kalman_mean = args[0]
        kalman_var = args[1]
        parameters = args[2]
        v = 1/particles[0, :]
        s0 = parameters[2]
        s = s0 * np.ones(len(v))

        var = kalman_var + v + s
        likelihood = np.exp(-1 / (2 * var) * ((observation - kalman_mean) ** 2)) / np.sqrt(2 * np.pi * var)
        return likelihood

    @staticmethod
    def _observation_likelihood_vol_lesion(particles, observation, args):
        kalman_mean = args[0]
        kalman_var = args[1]
        parameters = args[2]
        s = 1/particles[0, :]
        v0 = parameters[3]
        v = v0*np.ones(len(s))

        var = kalman_var + v + s
        likelihood = np.exp(-1 / (2 * var) * ((observation - kalman_mean) ** 2)) / np.sqrt(2 * np.pi * var)
        return likelihood

    @staticmethod
    def _kalman_update(particles, observation, kalman_mean, kalman_var, args):
        s = 1/particles[0, :]
        v = 1/particles[1, :]

        kalman_gain = (kalman_var + v) / (kalman_var + v + s)
        kalman_mean = kalman_mean + kalman_gain * (observation - kalman_mean)
        kalman_var = (1 - kalman_gain) * (kalman_var + v)
        return kalman_mean, kalman_var, kalman_gain

    @staticmethod
    def _kalman_update_stc_lesion(particles, observation, kalman_mean, kalman_var, args):
        v = 1/particles[0, :]
        s0 = args[2]
        s = s0 * np.ones(len(v))

        kalman_gain = (kalman_var + v) / (kalman_var + v + s)
        kalman_mean = kalman_mean + kalman_gain * (observation - kalman_mean)
        kalman_var = (1 - kalman_gain) * (kalman_var + v)
        return kalman_mean, kalman_var, kalman_gain

    @staticmethod
    def _kalman_update_vol_lesion(particles, observation, kalman_mean, kalman_var, args):
        s = 1/particles[0, :]
        v0 = args[3]
        v = v0 * np.ones(len(s))

        kalman_gain = (kalman_var + v) / (kalman_var + v + s)
        kalman_mean = kalman_mean + kalman_gain * (observation - kalman_mean)
        kalman_var = (1 - kalman_gain) * (kalman_var + v)
        return kalman_mean, kalman_var, kalman_gain

    def run(self, observations):
        num_observations = len(observations)
        num_particles = self.num_particles
        kalman_mean = np.zeros(num_particles)
        kalman_var = self.x0_unc * np.ones(num_particles)
        parameters = [self.lambda_stc, self.lambda_vol, self.init_stc, self.init_vol]

        initial_state = self.initial_state
        initial_state = np.tile(initial_state, (1, num_particles))
        pf = particle_filter.ParticleFilter(self.transition_sample, self.observation_likelihood, initial_state, self.resampling)

        vol = np.zeros(num_observations)
        stc = np.zeros(num_observations)
        val = np.zeros(num_observations)
        lr = np.zeros(num_observations)
        state = np.zeros((initial_state.shape[0], num_observations))
        for t in range(num_observations):
            state[:, t] = pf.predict(parameters)
            pf.correct(observations[t], [kalman_mean, kalman_var, parameters])
            (kalman_mean, kalman_var, kalman_gain) = self.kalman_update(pf.particles, observations[t], kalman_mean,
                                                                        kalman_var, parameters)
            val[t] = pf.weights @ kalman_mean
            lr[t] = pf.weights @ kalman_gain

        if all(self.model_type):
            stc = 1/state[0, :]
            vol = 1/state[1, :]
        elif self.model_type[0] is False:
            vol = 1/state[0, :]
            stc = self.init_stc * np.ones((1, num_observations))
        elif self.model_type[1] is False:
            stc = 1/state[0, :]
            vol = self.init_vol * np.ones((1, num_observations))

        return lr, stc, vol, val


class LearningModelGaussian:
    def __init__(self, sigma_stc, sigma_vol, init_stc=1, init_vol=1, x0_unc=100, num_particles=100,
                 resampling_method='systematic'):
        # sigma_stc and sigma_vol are variance parameters for diffusion of stochasticity and volatility, respectively.
        # Their role is analogous to that of lambda_stc and lambda_vol in the previous class.
        # These parameters should be positive (although it is possible to set one of them as 0)
        # In practice, it is better to set these parameters in the range [0 0.5]
        # If either of these parameters are 0, this class creates the corresponding lesion model.
        # init_stc and init_vol are initial values (i.e. 0th trial) for stochasticity and volatility, respectively.
        # They both should be positive.
        # x0_unc is the initial value for Kalman variance. It should be positive.
        # num_particles determines the number of particles for the particle filter.
        # resampling determines the resampling method for the particle filter: systematic or binomial
        # The main method of this class is run(observations), which runs the constructed model for a timeseries observations

        self.sigma_stc = sigma_stc
        self.sigma_vol = sigma_vol
        self.init_stc = init_stc
        self.init_vol = init_vol
        self.x0_unc = x0_unc
        self.num_particles = num_particles
        self.resampling = resampling_method

        if sigma_stc == 0 and sigma_vol == 0:
            raise ValueError('Both sigma parameters are None!')

        if sigma_stc != 0 and sigma_vol != 0:
            self.model_name = 'healthy'
            self.model_type = [True, True]
            self.transition_sample = self._transition_sample
            self.observation_likelihood = self._observation_likelihood
            self.kalman_update = self._kalman_update
            initial_state = np.zeros((2, 1))
            initial_state[0] = np.log(self.init_stc)
            initial_state[1] = np.log(self.init_vol)
            self.initial_state = initial_state
        elif sigma_stc == 0:
            self.model_name = 'stochasticity lesion'
            self.model_type = [False, True]
            self.transition_sample = self._transition_sample_stc_lesion
            self.observation_likelihood = self._observation_likelihood_stc_lesion
            self.kalman_update = self._kalman_update_stc_lesion
            initial_state = np.zeros((1, 1))
            initial_state[0] = np.log(self.init_vol)
            self.initial_state = initial_state
            print('Model is %s')
        elif sigma_vol == 0:
            self.model_name = 'volatility lesion'
            self.model_type = [True, False]
            self.transition_sample = self._transition_sample_vol_lesion
            self.observation_likelihood = self._observation_likelihood_vol_lesion
            self.kalman_update = self._kalman_update_vol_lesion
            initial_state = np.zeros((1, 1))
            initial_state[0] = np.log(self.init_stc)
            self.initial_state = initial_state

        print('Model is %s' % (self.model_name,))

    @staticmethod
    def _transition_sample(particles, parameters):
        num_particles = particles.shape[1]

        s = particles[0, :]
        sigma_s = parameters[0]
        s = s + sigma_s * np.random.normal(0, 1, num_particles)

        v = particles[1, :]
        sigma_v = parameters[1]
        v = v + sigma_v * np.random.normal(0, 1, num_particles)

        particles[0, :] = s
        particles[1, :] = v
        return particles

    @staticmethod
    def _transition_sample_stc_lesion(particles, parameters):
        num_particles = particles.shape[1]
        v = particles[0, :]
        sigma_v = parameters[1]
        v = v + sigma_v * np.random.normal(0, 1, num_particles)

        particles[0, :] = v
        return particles

    @staticmethod
    def _transition_sample_vol_lesion(particles, parameters):
        num_particles = particles.shape[1]
        s = particles[0, :]
        sigma_s = parameters[0]
        s = s + sigma_s * np.random.normal(0, 1, num_particles)

        particles[0, :] = s
        return particles

    @staticmethod
    def _observation_likelihood(particles, observation, args):
        kalman_mean = args[0]
        kalman_var = args[1]
        s = np.exp(particles[0, :])
        v = np.exp(particles[1, :])
        var = kalman_var + v + s
        likelihood = np.exp(-1 / (2 * var) * ((observation - kalman_mean) ** 2)) / np.sqrt(2 * np.pi * var)
        return likelihood

    @staticmethod
    def _observation_likelihood_stc_lesion(particles, observation, args):
        kalman_mean = args[0]
        kalman_var = args[1]
        parameters = args[2]
        v = np.exp(particles[0, :])
        s0 = parameters[2]
        s = s0 * np.ones(len(v))

        var = kalman_var + v + s
        likelihood = np.exp(-1 / (2 * var) * ((observation - kalman_mean) ** 2)) / np.sqrt(2 * np.pi * var)
        return likelihood

    @staticmethod
    def _observation_likelihood_vol_lesion(particles, observation, args):
        kalman_mean = args[0]
        kalman_var = args[1]
        parameters = args[2]
        s = np.exp(particles[0, :])
        v0 = parameters[3]
        v = v0*np.ones(len(s))

        var = kalman_var + v + s
        likelihood = np.exp(-1 / (2 * var) * ((observation - kalman_mean) ** 2)) / np.sqrt(2 * np.pi * var)
        return likelihood

    @staticmethod
    def _kalman_update(particles, observation, kalman_mean, kalman_var, args):
        s = np.exp(particles[0, :])
        v = np.exp(particles[1, :])

        kalman_gain = (kalman_var + v) / (kalman_var + v + s)
        kalman_mean = kalman_mean + kalman_gain * (observation - kalman_mean)
        kalman_var = (1 - kalman_gain) * (kalman_var + v)
        return kalman_mean, kalman_var, kalman_gain

    @staticmethod
    def _kalman_update_stc_lesion(particles, observation, kalman_mean, kalman_var, args):
        v = np.exp(particles[0, :])
        s0 = args[2]
        s = s0 * np.ones(len(v))

        kalman_gain = (kalman_var + v) / (kalman_var + v + s)
        kalman_mean = kalman_mean + kalman_gain * (observation - kalman_mean)
        kalman_var = (1 - kalman_gain) * (kalman_var + v)
        return kalman_mean, kalman_var, kalman_gain

    @staticmethod
    def _kalman_update_vol_lesion(particles, observation, kalman_mean, kalman_var, args):
        s = np.exp(particles[0, :])
        v0 = args[3]
        v = v0 * np.ones(len(s))

        kalman_gain = (kalman_var + v) / (kalman_var + v + s)
        kalman_mean = kalman_mean + kalman_gain * (observation - kalman_mean)
        kalman_var = (1 - kalman_gain) * (kalman_var + v)
        return kalman_mean, kalman_var, kalman_gain

    def run(self, observations):
        num_observations = len(observations)
        num_particles = self.num_particles
        kalman_mean = np.zeros(num_particles)
        kalman_var = self.x0_unc * np.ones(num_particles)
        parameters = [self.sigma_stc, self.sigma_vol, self.init_stc, self.init_vol]

        initial_state = self.initial_state
        initial_state = np.tile(initial_state, (1, num_particles))
        pf = particle_filter.ParticleFilter(self.transition_sample, self.observation_likelihood, initial_state, self.resampling)

        vol = np.zeros(num_observations)
        stc = np.zeros(num_observations)
        val = np.zeros(num_observations)
        lr = np.zeros(num_observations)
        state = np.zeros((initial_state.shape[0], num_observations))
        for t in range(num_observations):
            state[:, t] = pf.predict(parameters)
            pf.correct(observations[t], [kalman_mean, kalman_var, parameters])
            (kalman_mean, kalman_var, kalman_gain) = self.kalman_update(pf.particles, observations[t], kalman_mean,
                                                                        kalman_var, parameters)
            val[t] = pf.weights @ kalman_mean
            lr[t] = pf.weights @ kalman_gain

        if all(self.model_type):
            stc = np.exp(state[0, :])
            vol = np.exp(state[1, :])
        elif self.model_type[0] is False:
            vol = np.exp(state[0, :])
            stc = self.init_stc * np.ones((1, num_observations))
        elif self.model_type[1] is False:
            stc = np.exp(state[0, :])
            vol = self.init_vol * np.ones((1, num_observations))

        return lr, stc, vol, val


