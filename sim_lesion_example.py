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

num_sim = 100
num_observations = 200

true_vol = np.array([0.5, 0.5, 1.5, 1.5])
true_stc = np.array([1, 3, 1, 3])
label_vol = ['small', 'small', 'large', 'large']
label_stc = ['small', 'large', 'small', 'large']

lambda_stc = [0.1, 0, 0.1]
lambda_vol = [0.1, 0.1, 0]
init_vol = np.mean(true_vol)
init_stc = np.mean(true_stc)

dfs = []
model_names = []
for m in range(3):
    # if lambda_stc is zero, this creates a stochasticity lesion model
    # if lambda_vol is zero, this creates a volatility lesion model
    # otherwise this creates a healthy model
    model = learning_models.LearningModel(lambda_stc[m], lambda_vol[m], init_stc, init_vol, 100, 100)

    lrs = np.zeros((num_sim, 4))
    vols = np.zeros((num_sim, 4))
    stcs = np.zeros((num_sim, 4))
    for k in range(len(true_stc)):
        for i in range(num_sim):
            observations = generate_data(num_observations, true_vol[k], true_stc[k])
            (lr, stc, vol, _) = model.run(observations)
            lrs[i, k] = np.mean(lr[-20:])
            stcs[i, k] = np.mean(stc[-20:])
            vols[i, k] = np.mean(vol[-20:])

    true_vols = np.tile(true_vol, (num_sim, 1))
    true_stcs = np.tile(true_stc, (num_sim, 1))

    lr = np.reshape(lrs, lrs.size)
    vol = np.reshape(vols, vols.size)
    stc = np.reshape(stcs, stcs.size)
    true_vols = np.reshape(true_vols, stcs.size)
    true_stcs = np.reshape(true_stcs, stcs.size)

    d = {'Learning rate': lr, 'Stochasticty': stc, 'Volatility': vol, 'True volatility': true_vols,
         'True stochasticty': true_stcs}
    df = pd.DataFrame(data=d)
    df['True volatility'] = df['True volatility'].apply(lambda x: 'Large' if x == np.amax(true_vol) else 'Small')
    df['True stochasticty'] = df['True stochasticty'].apply(lambda x: 'Large' if x == np.amax(true_stc) else 'Small')
    model_names.append(model.model_name)

    dfs.append(df)

plt.figure(figsize=(15, 8), dpi=80)
# sns.set(font_scale = 1.2)

for m in range(3):
    plt.subplot(2, 3, m + 1)
    ax = sns.barplot(x="True volatility", y="Learning rate", data=dfs[m], ci=None, hue="True stochasticty",
                     hue_order=['Small', 'Large'], alpha=0.8)
    ax.set_xticklabels(['Small', 'Large'])
    ax.set(ylim=(0, 1))
    ax.set_title(model_names[m])
    ax.set_ylim([0, 0.75])
    if m != 1:
        plt.legend([], [], frameon=False)

plt.subplot(2, 3, 5)
ax = sns.barplot(x="True volatility", y="Volatility", data=dfs[1], ci=None, hue="True stochasticty",
                 hue_order=['Small', 'Large'], alpha=0.8)
ax.set_xticklabels(['Small', 'Large'])
ax.set_ylim([0, 2.5])
plt.legend([], [], frameon=False)

plt.subplot(2, 3, 6)
ax = sns.barplot(x="True volatility", y="Stochasticty", data=dfs[2], ci=None, hue="True stochasticty",
                 hue_order=['Small', 'Large'], alpha=0.8)
ax.set_xticklabels(['Small', 'Large'])
ax.set_ylim([0, 3.5])
plt.legend([], [], frameon=False)

plt.show()
