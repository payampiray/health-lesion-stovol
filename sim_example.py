import numpy as np
import learning_models

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_data(num_observations, true_vol, true_stc):
    x = np.zeros(num_observations)
    for t in range(1, num_observations):
        x[t] = x[t - 1] + np.sqrt(true_vol) * np.random.normal()
    y = x + np.sqrt(true_stc) * np.random.normal(0, 1, num_observations)
    return y


np.random.seed(0)
true_vol = np.array([0.5, 0.5, 1.5, 1.5])
true_stc = np.array([1, 3, 1, 3])
num_sim = 100
num_observations = 200

lambda_stc = 0.1
lambda_vol = 0.1
init_vol = np.mean(true_vol)
init_stc = np.mean(true_stc)
model = learning_models.LearningModel(lambda_stc, lambda_vol, init_stc, init_vol, 100, 100)

lrs_last20 = np.zeros((num_sim, 4))
stcs_last20 = np.zeros((num_sim, 4))
vols_last20 = np.zeros((num_sim, 4))
lr_mean = np.zeros((num_observations, 4))
stc_mean = np.zeros((num_observations, 4))
vol_mean = np.zeros((num_observations, 4))
for k in range(4):
    lrs = np.zeros((num_observations, num_sim))
    stcs = np.zeros((num_observations, num_sim))
    vols = np.zeros((num_observations, num_sim))
    for i in range(num_sim):
        observations = generate_data(num_observations, true_vol[k], true_stc[k])
        (lr, stc, vol, _) = model.run(observations)
        lrs_last20[i, k] = np.mean(lr[-20:])
        stcs_last20[i, k] = np.mean(stc[-20:])
        vols_last20[i, k] = np.mean(vol[-20:])
        lrs[:, i] = lr
        stcs[:, i] = stc
        vols[:, i] = vol
    lr_mean[:, k] = np.mean(lrs, 1)
    stc_mean[:, k] = np.mean(stcs, 1)
    vol_mean[:, k] = np.mean(vols, 1)

signals = [lr_mean[:, 0:2], lr_mean[:, 2:4], vol_mean[:, 0:2], vol_mean[:, 2:4], stc_mean[:, 0:2], stc_mean[:, 2:4]]
titles = ['Learning rate', 'Learning rate', 'Volatility', 'Volatility', 'Stochasticity', 'Stochasticity']
ylims = [[0.2, 0.8], [0.2, 0.8], [0, 1.8], [0, 1.8], [0, 3.5], [0, 3.5]]

plt.figure(figsize=(15, 6), dpi=80)
for m in range(6):
    ax = plt.subplot(3, 2, m + 1)
    ax.plot(signals[m])
    ax.set_ylabel(titles[m])
    ax.set_ylim(ylims[m])
    if m == 0:
        ax.set_title('Small true volatility')
    elif m == 1:
        ax.set_title('Large true volatility')
    elif m == 2:
        h = ax.legend(['Small', 'Large'], frameon=False, title='True stochasticity')
    elif m > 3:
        ax.set_xlabel('Trial')

true_vols = np.tile(true_vol, (num_sim, 1))
true_stcs = np.tile(true_stc, (num_sim, 1))

lr_last20 = np.reshape(lrs_last20, lrs_last20.size)
vol_last20 = np.reshape(vols_last20, vols_last20.size)
stc_last20 = np.reshape(stcs_last20, stcs_last20.size)
true_vols = np.reshape(true_vols, stcs_last20.size)
true_stcs = np.reshape(true_stcs, stcs_last20.size)

d = {'Learning rate': lr_last20, 'Stochasticty': stc_last20, 'Volatility': vol_last20, 'True volatility': true_vols,
     'True stochasticty': true_stcs}
df = pd.DataFrame(data=d)
df['True volatility'] = df['True volatility'].apply(lambda x: 'Large' if x == np.amax(true_vol) else 'Small')
df['True stochasticty'] = df['True stochasticty'].apply(lambda x: 'Large' if x == np.amax(true_stc) else 'Small')

plt.figure(figsize=(15, 4), dpi=80)
# sns.set(font_scale = 1.2)

plt.subplot(1, 3, 1)
ax = sns.barplot(x="True volatility", y="Learning rate", data=df, ci=None, hue="True stochasticty",
                 hue_order=['Small', 'Large'], alpha=0.8)
ax.set_xticklabels(['Small', 'Large'])
ax.set(ylim=(0, 1))

plt.subplot(1, 3, 2)
ax = sns.barplot(x="True volatility", y="Volatility", data=df, ci=None, hue="True stochasticty",
                 hue_order=['Small', 'Large'], alpha=0.8)
ax.set_xticklabels(['Small', 'Large'])
plt.legend([], [], frameon=False)

plt.subplot(1, 3, 3)
ax = sns.barplot(x="True volatility", y="Stochasticty", data=df, ci=None, hue="True stochasticty",
                 hue_order=['Small', 'Large'], alpha=0.8)
ax.set_xticklabels(['Small', 'Large'])
plt.legend([], [], frameon=False)

plt.show()
