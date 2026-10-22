# Scoring

FORGE-Bench scores each video on six axes, each from 0 to 100:

| Symbol | Axis |
|---|---|
| $L$ | Industrial logic and fact alignment |
| $G$ | Geometric integrity |
| $P$ | Physical plausibility |
| $T$ | Temporal consistency |
| $R$ | Reference and motion fidelity |
| $U$ | Application usefulness |

CV evidence can cap a judge score when its operator is eligible: it has a recognized name, valid output, `used_for_axis_cap=true`, and confidence at least 0.70. The current caps are:

| Evidence | Affected axis and maximum |
|---|---|
| Global regeneration | $R \le 35$, $T \le 45$ |
| Large nonlocal change | $R \le 50$ |
| Abrupt temporal break | $T \le 35$ |
| Late abrupt break | $T \le 25$ |
| Rigid joint drift | $G \le 45$ |

Let $C_i$ be the percentage of required events observed in video $i$, and let the primed axis scores include any eligible caps. The sample and model scores are

$$S_i = \frac{C_i}{100} \cdot \frac{L_i' + G_i' + P_i' + T_i' + R_i' + U_i'}{6}, \qquad S_{model} = \frac{1}{N}\sum_{i=1}^{N} S_i.$$

Required events have equal weight. Axis means, event coverage, task success, visual quality, and failure modes are reported separately. The model score uses the mean of sample scores.

Confidence intervals use a scene-level cluster bootstrap with 1,000 resamples and seed 1729. The requested task manifest defines the evaluation population. Missing videos, axis scores, or event coverage appear as incomplete results.

The executable policy is in `scoring/forge_v4_config.json` (version `forge-video-5plus1-event-gated-mean-v5.0`).
