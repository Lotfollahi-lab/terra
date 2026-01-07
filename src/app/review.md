# BioFlow Review

## Summary

This paper proposes **BioFlow**, a **histology-conditioned conditional flow-matching (CFM)** model for predicting **spatial transcriptomics gene expression** from **H&E whole-slide image patches**. The central claim is that existing generative approaches (diffusion, flow matching) can traverse **negative-valued expression regions** during their continuous trajectories, which is biologically invalid for count-like expression data. BioFlow introduces a **support-preserving velocity-field reparameterization** intended to guarantee **non-negativity throughout the entire generative trajectory**, and pairs it with a **Transformer-based multimodal velocity network** that models image and gene tokens with decomposed cross-modal attention. Experiments on HER2ST, PRAD, and READ report improved PCC (especially PCC(All)) and improved efficiency compared to STFlow/TRIPLEX/MERGE, with a separate comparison to STEM using reported official results due to cost.

---

## Strengths

- **Clear and relevant motivation:** Gene expression is non-negative and sparse; the paper highlights a plausible failure mode where unconstrained continuous generation explores negative regions (Fig. 1 and Sec. 2.2).

- **Simple, targeted technical idea:** Enforce non-negativity via a velocity-field constraint derived from the numerical update condition (Eqs. 7–9), rather than post-hoc clamping at the end.

- **Strong quantitative results on reported metric:** BioFlow improves **PCC(All)** on all three datasets vs. the best baseline (Table 1), and shows improved stability from easy → hard genes.

- **Efficiency focus and protocol hygiene:** The paper explicitly calls out test-patient leakage in prior early stopping/model selection and claims to retrain baselines under a corrected protocol (Sec. 4.1).

- **Ablations included:** Impact of the non-negative constraint (Table 4) and of the Transformer velocity network (Table 5) are reported and directionally consistent with the narrative.

---

## Major Weaknesses

### 1. Core "guarantee" is not convincingly established and appears problematic as written

The reparameterization in Eq. (9) uses $-x_t / \Delta t + \text{softplus}(\cdot)$ and is justified via the discrete update inequality in Eq. (8). However, plugging Eq. (9) into the Euler update Eq. (7) yields:

$$
x_{t+\Delta t} = x_t + \Delta t \left( -\frac{x_t}{\Delta t} + \text{softplus}(\cdot) \right) = \Delta t \cdot \text{softplus}(\cdot)
$$

which removes the additive carry-over of $x_t$. This is a major conceptual issue for an ODE-based flow model and makes the "support-preserving probability path" claim unclear without additional justification (e.g., solver choice, step-size dependence, continuous-time interpretation, and confirmation that this is exactly what is implemented).

### 2. Inconsistency / ambiguity about the prior and where negativity arises

Training describes sampling $x_0$ from "e.g., Gaussian noise" (Sec. 3.2), which would make the *linear interpolation path* $x_t = (1-t)x_0 + t x_1$ negative for small $t$ whenever $x_0$ has negative components (Sec. 3.2–3.3). Yet inference (Algorithm 2) states a **ZINB prior**. This mismatch undermines the narrative that BioFlow preserves non-negativity "throughout the entire trajectory," and makes it difficult to assess whether the negativity problem is primarily from the *prior/path construction* or from the *learned vector field during integration*.

### 3. Missing strong controls for the biological-validity hypothesis

The paper argues that trajectory-level non-negativity is critical, but does not provide comparisons to simpler alternatives (e.g., softplus/exp output parameterization, log-domain modeling, stepwise projection/clamping each integration step). Without these baselines, it is hard to isolate whether the gains come from "trajectory support preservation" versus easier fixes.

### 4. Evaluation is narrow for a "generative/stochastic" framing

The method is motivated as generative and stochastic (many-to-many mapping), but evaluation is dominated by **PCC** rankings (Sec. 4.1–4.2). There is limited analysis of uncertainty/diversity, distributional fit to count properties (e.g., zero inflation / NB behavior), or downstream biological utility beyond correlation.

---

## Minor Weaknesses

- **Efficiency claims in the abstract appear overstated relative to reported training times.**  
  The abstract claims "4× to 100× faster than STFlow" (p.1), but the concrete minutes reported for STFlow vs BioFlow (e.g., 396 vs 122 on HER2ST; 60 vs 18 on PRAD; 17 vs 7 on READ) correspond more to ~2–3× in those settings (Sec. 4.3). The "100×" regime is not clearly supported in the main reported comparisons.

- **STEM comparison is not retrained under the corrected protocol.**  
  The paper reports official STEM results due to prohibitive cost (Sec. 4.4), which is understandable, but weakens the fairness claim given the paper's emphasis on evaluation leakage corrections (Sec. 4.1).

- **Metric definitions could be clearer in the main text.**  
  PCC(H), PCC(M), PCC(All) are described qualitatively (Sec. 4.1), but the exact gene-count cutoffs are not explicitly stated in the main table narrative (Table 1).

- **Small patient counts in PRAD/READ increase variance risk.**  
  PRAD and READ have only 2 patients (Table 2), so results can be high variance; reporting confidence intervals or fold variability would strengthen claims.

---

## Justification

**Recommendation: Weak Reject (5/10)**

The paper tackles a meaningful and domain-relevant issue (non-negativity / biological validity) and reports strong PCC gains with a seemingly efficient architecture. However, the central technical contribution—the velocity reparameterization claimed to *guarantee* non-negative trajectories—appears insufficiently justified and possibly problematic as written, especially given the dependence on $\Delta t$ and the resulting update behavior under Euler integration.

Additionally, ambiguity around the prior (Gaussian vs ZINB) and lack of simpler non-negativity baselines make it hard to attribute improvements to the claimed mechanism.

With clearer mathematical grounding (or correction of the formulation), reconciled training/inference priors, and stronger ablations/controls, this could become a solid contribution.
