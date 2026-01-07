# BioFlow Review

## Summary

This paper proposes **BioFlow**, a **histology-conditioned conditional flow-matching (CFM)** model for predicting **spatial transcriptomics gene expression** from **H&E whole-slide image patches**. The central claim is that existing generative approaches (diffusion, flow matching) can traverse **negative-valued expression regions** during their continuous trajectories, which is biologically invalid for count-like expression data. BioFlow introduces a **support-preserving velocity-field reparameterization** intended to guarantee **non-negativity throughout the entire generative trajectory**, and pairs it with a **Transformer-based multimodal velocity network** that models image and gene tokens with decomposed cross-modal attention. Experiments on HER2ST, PRAD, and READ report improved PCC (especially PCC(All)) and improved efficiency compared to STFlow/TRIPLEX/MERGE, with a separate comparison to STEM using reported official results due to cost.

## Strengths

1. **Clear and relevant motivation:** Gene expression is non-negative and sparse; the paper highlights a plausible failure mode where unconstrained continuous generation explores negative regions (Fig. 1 and Sec. 2.2).

2. **Simple, targeted technical idea:** Enforce non-negativity via a velocity-field constraint derived from the numerical update condition (Eqs. 7 to 9), rather than post-hoc clamping at the end.

3. **Strong quantitative results on reported metric:** BioFlow improves **PCC(All)** on all three datasets vs. the best baseline (Table 1), and shows improved stability from easy to hard genes.

4. **Efficiency focus and protocol hygiene:** The paper explicitly calls out test-patient leakage in prior early stopping/model selection and claims to retrain baselines under a corrected protocol (Sec. 4.1).

5. **Ablations included:** Impact of the non-negative constraint (Table 4) and of the Transformer velocity network (Table 5) are reported and directionally consistent with the narrative.

## Major Weaknesses

### 1. Core "guarantee" is not convincingly established and appears problematic as written

The problem is not that BioFlow cannot keep outputs non-negative. It is that the paper's *claimed guarantee* is easy to misunderstand (and possibly wrong as stated) because it quietly depends on **how inference is numerically integrated** and **what Δt actually means**.

#### 1.1 The guarantee is a property of a specific *discrete update*, not the ODE itself

The authors derive a condition from **forward Euler**:

$$
x_{k+1} = x_k + h \cdot v(x_k, t_k) \ge 0 \iff v \ge -x_k / h
$$

That is a *discretization* statement. If you change solver (RK4, adaptive), the intermediate stages can go negative even if Euler would not. So the "trajectory stays non-negative" claim only truly holds under specific solver/step assumptions that are not explicitly stated.

#### 1.2 If Δt equals the solver step h, the update "cancels" the state

With their parameterization v̂ = −x / Δt + softplus(·) and Euler with h = Δt:

$$
x_{k+1} = x_k + \Delta t \left( -\frac{x_k}{\Delta t} + \text{softplus}(\cdot) \right) = \Delta t \cdot \text{softplus}(\cdot)
$$

This means each step is basically "overwrite with a positive prediction scaled by Δt," **not** a normal residual flow update. It also makes the behavior extremely sensitive to the choice/number of steps. This raises correctness/clarity alarms for an ODE-based flow model and makes the "support-preserving probability path" claim unclear.

#### 1.3 If Δt is *not* the solver step, then positivity becomes step-size-conditional

If solver step is h and Δt is a fixed constant:

$$
x_{k+1} = \left(1 - \frac{h}{\Delta t}\right) x_k + h \cdot \text{softplus}(\cdot)
$$

Now positivity holds **if and only if** 0 ≤ h ≤ Δt. So the "guarantee" is: *use Euler (or a positivity-preserving method) with step bounded by Δt*. That is valid, but it is not the broad "structural" guarantee the paper language often suggests unless they state these conditions explicitly.

#### 1.4 The "fundamental failure mode" framing overstates the problem

The paper calls negativity a "fundamental failure mode" of other generative models, but it is mostly a modeling/implementation choice. Negativity in baselines often comes from: (1) choosing a base distribution with negative support (Gaussian), (2) unconstrained velocity fields, and (3) solver overshoot.

But there are standard fixes (log-space modeling, exp/softplus state parameterization, projection/clamping each step). If the paper does not compare to these simpler alternatives, it is hard to argue BioFlow is uniquely necessary.

#### Summary of this issue

BioFlow's positivity trick can work, but the paper's *guarantee* is **ambiguous and potentially misleading** unless they precisely define Δt, specify the solver, and prove the guarantee under the exact inference procedure (and step-size conditions) they use. Without this clarity, it remains unclear whether the formulation is "just unclear writing" or a real technical mistake.

### 2. Inconsistency / ambiguity about the prior and where negativity arises

Training describes sampling x₀ from "e.g., Gaussian noise" (Sec. 3.2), which would make the *linear interpolation path* xₜ = (1−t)x₀ + t x₁ negative for small t whenever x₀ has negative components (Sec. 3.2 to 3.3). Yet inference (Algorithm 2) states a **ZINB prior**. This mismatch undermines the narrative that BioFlow preserves non-negativity "throughout the entire trajectory," and makes it difficult to assess whether the negativity problem is primarily from the *prior/path construction* or from the *learned vector field during integration*.

### 3. Missing strong controls for the biological-validity hypothesis

The paper argues that trajectory-level non-negativity is critical, but does not provide comparisons to simpler alternatives (e.g., softplus/exp output parameterization, log-domain modeling, stepwise projection/clamping each integration step). Without these baselines, it is hard to isolate whether the gains come from "trajectory support preservation" versus easier fixes.

### 4. Evaluation is narrow for a "generative/stochastic" framing

The method is motivated as generative and stochastic (many-to-many mapping), but evaluation is dominated by **PCC** rankings (Sec. 4.1 to 4.2). There is limited analysis of uncertainty/diversity, distributional fit to count properties (e.g., zero inflation / NB behavior), or downstream biological utility beyond correlation.

## Minor Weaknesses

1. **Efficiency claims in the abstract appear overstated relative to reported training times.** The abstract claims "4× to 100× faster than STFlow" (p.1), but the concrete minutes reported for STFlow vs BioFlow (e.g., 396 vs 122 on HER2ST; 60 vs 18 on PRAD; 17 vs 7 on READ) correspond more to approximately 2 to 3× in those settings (Sec. 4.3). The "100×" regime is not clearly supported in the main reported comparisons.

2. **STEM comparison is not retrained under the corrected protocol.** The paper reports official STEM results due to prohibitive cost (Sec. 4.4), which is understandable, but weakens the fairness claim given the paper's emphasis on evaluation leakage corrections (Sec. 4.1).

3. **Metric definitions could be clearer in the main text.** PCC(H), PCC(M), PCC(All) are described qualitatively (Sec. 4.1), but the exact gene-count cutoffs are not explicitly stated in the main table narrative (Table 1).

4. **Small patient counts in PRAD/READ increase variance risk.** PRAD and READ have only 2 patients (Table 2), so results can be high variance; reporting confidence intervals or fold variability would strengthen claims.

## Justification

**Recommendation: Weak Reject (2/6)**

The paper tackles a meaningful and domain-relevant issue (non-negativity / biological validity) and reports strong PCC gains with a seemingly efficient architecture. However, the central technical contribution (the velocity reparameterization claimed to *guarantee* non-negative trajectories) appears insufficiently justified and possibly problematic as written, especially given the dependence on Δt and the resulting update behavior under Euler integration.

Additionally, ambiguity around the prior (Gaussian vs ZINB) and lack of simpler non-negativity baselines make it hard to attribute improvements to the claimed mechanism.

