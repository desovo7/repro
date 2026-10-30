# RePro v8.3.1 — Codex Implementation Specification (verl-agent / ALFWorld)

> **Purpose of this Document**: This is a self-contained engineering specification for implementing **RePro v8.3.1** (including **Tier 1A, Tier 1B, and Tier 2**) on top of the `verl-agent` GiGPO baseline for ALFWorld, strictly calibrated against the full 150-iteration Step 0 audit (`step0_training_audit.json` & `final_binding_9488452854dd_a800.json`).
>
> **Empirical Baseline from Step 0 Audit (150 Iterations, 2,400 Groups, 19,200 Trajectories, 594,382 Actions)**:
>
> 1. **Anchor Group Sparsity & Mandatory Fallback**: Aggregate P25/P50 remained `1 / 2`, **but `8 / 150` iterations dropped to** $\text{P50} = 1$ **under full de-aliasing and triggered the fallback**, and the singleton anchor group rate ($\vert{}G(z)\vert{} = 1$) increased from **`37.38%` to `44.36%` (`+6.98 pp`)**. Therefore, **the per-iteration dynamic fallback (`Critical-Only De-Aliasing` when** $\text{P50}_{\text{full}} < 2$**) MUST be strictly retained**.
>
> 2. **Conditional Sign-Change Rate**: Within the pre-filtered subset of empty-handed arrivals at goal/tool receptacles in mixed ($\text{AF}=0$) groups, de-aliasing changed the advantage sign on **`2,596 / 41,998 = 6.18%`** of actions. *(Note: This is a conditional sign-change sensitivity metric within that pre-selected subset, not a global rate across all 594,382 actions).*
>
> 3. **All-Failed (AF) Group Training Share**: Across 150 iterations, AF groups account for **`566 / 2,400 = 23.58%`** of task groups.
>
> 4. **Regression Sanity Check**: Across the 45 audited evaluation trajectories (1,832 steps), `14 / 14` winning terminal steps had `0` false penalties (necessary regression pass on audited cases; intermediate step effects require training validation).
>
> 5. **Measured Latency & GC Tail**: Static regexes and text caching are already active. Mean runtime is **`0.086s / iter`**, max **`0.130s / iter`** (with `46 / 150` iterations $> 0.10\text{s}$ due to `~0.125s` Python GC pauses). At `0.03%–0.05%` of a `180–300s` training iteration, this runtime profile is accepted without hard $<0.10\text{s}$ blocking.

---

## 1. Rollout & Ablation Execution Order (Strict Phased Gate)

Implement and run in the following modular order (controlled via config flags):

1. **Step 0 (`run_step0_audit.py`)**: \[COMPLETED — Baseline Logged\]
   * Logged 594,382 actions: `8/150` fallback triggers, singleton rate `37.38% -> 44.36%`, conditional sign-change `2,596/41,998 (6.18%)`, mean latency `0.086s/iter` (max `0.130s`).

2. **Tier 1A (`enable_tier1a=True`)**:
   * **T1-1**: All-6-family instance-locked & conflict-aware progress state machine $p_t \in \{-1, 0, \dots, K_f\}$.
   * **T1-2**: De-aliased GiGPO step anchor key `AnchorKey = (norm(obs_t), p_t, b_nav_t)` **with mandatory per-iteration dynamic fallback (`Critical-Only De-Aliasing` whenever per-iteration** $\text{P50}_{\text{full}} < 2$**)**.

3. **Tier 1B (`enable_tier1b=True`)**:
   * **T1-3**: All-failed group ($\text{AF}_x = 1$, covering `23.58%` of training groups) step-level credit combining potential difference $\Delta \Phi$, candidate-aware navigation directional signal $b^{\text{nav}}_{t+1}$ (unseen exploration $e_{t+1}$ vs. known-candidate attraction $a_{t+1}$), and generalized No-Op / state-invariant action truncation ($\lambda_t = 0, \delta_t = -c_{\text{noop}}$).
   * **Controlled Ablation Requirement**: Run `Tier 1A only` and `Tier 1A + Tier 1B` under unified hardware/TP and frozen task manifests to separately test the causal contribution of $\text{AF}=0$ de-aliasing vs. $\text{AF}=1$ step credit.

4. **Tier 2 (`enable_tier2=True`)**:
   * **T2-1**: Replace the static prior $\Phi_{\text{prior}}(f, p_t)$ in $\text{AF}=1$ groups with the **Quotient Empirical Bayes Value Table** $\widetilde{V}(f, p_t, b^{\text{nav}}_t, b_{\text{step}}(t))$ (**180 nominal Cartesian cells reduced via two structural quotient rules to 96 reachable non-terminal cells**), estimated via first-visit Beta-Binomial shrinkage over a 4-iteration sliding window $[i-4, i-1]$ of $\text{AF}=0$ trajectories, guarded by isotonic monotonicity floors.
   * **T2-2**: Within-group, same-budget-bucket zero-mean centering of auxiliary advantages $\tilde{A}^{\text{aux}}_{j, t}$ to cancel task-level difficulty offsets and budget drift, plus per-iteration support & calibration logging.

---

## 2. Deterministic Trajectory State Tracker (`TrajectoryStateTracker`)

For each trajectory $\tau = (o_0, a_0, o_1, a_1, \dots, o_T)$ of task goal $g$ in family $f \in \{\text{simple}, \text{clean}, \text{heat}, \text{cool}, \text{look\_at}, \text{pick\_two}\}$, maintain a lightweight per-trajectory tracker updated sequentially at each step $t = 0, \dots, T$:

### 2.1 Parsed Task Metadata (Extracted Once at $t=0$ from Task Goal / Game File)

* `family` ($f$): Task family name (`pick_and_place_simple`, `pick_clean_then_place_in_recep`, `pick_heat_then_place_in_recep`, `pick_cool_then_place_in_recep`, `look_at_obj_in_light`, `pick_two_obj_and_place`).
* `target_type`: Canonical target object class string (e.g., `"pencil"`, `"soapbar"`, `"pan"`, `"mug"`, `"pillow"`, `"keychain"`, `"cd"`, `"newspaper"`).
  * **Strict Matching Rule**: An object instance string `<obj_cls> <id>` (e.g., `"pencil 1"`) matches `target_type` iff `<obj_cls> == target_type` (exact token match, so `"pen 1"` $\neq$ `"pencil"` and `"soapbottle 1"` $\neq$ `"soapbar"`).
* `goal_recep_type`: Canonical destination receptacle class string (e.g., `"desk"`, `"cart"`, `"countertop"`, `"coffeemachine"`, `"safe"`, `"sofa"`), or `None` for `look_at_obj_in_light`.
* `tool_recep_type`:
  * `"sinkbasin"` for `clean`;
  * `"microwave"` for `heat`;
  * `"fridge"` for `cool`;
  * `"desklamp"` for `look_at` (note: in ALFWorld, `desklamp` is a non-takeable fixture sitting on a host receptacle);
  * `None` for `simple` and `pick_two`.

### 2.2 Dynamic Per-Step State Variables (Updated at Step $t$)

1. `loc_t`: Current receptacle instance `<recep_cls> <id>` where the agent is located at step $t$ (updated whenever `o_t` matches `"You arrive at <recep>."` or `"You are in the middle of a room..."` $\to$ `None`).
2. `held_obj_t`: Object instance `<obj_cls> <id>` currently in the agent's hand (ALFWorld hand capacity is strictly 1):
   * Set `held_obj_t = "<obj>"` when `a_{t-1}` is `take <obj> from <recep>` and `o_t` is `"You pick up the <obj> from the <recep>."`;
   * Set `held_obj_t = None` when `a_{t-1}` is `move <obj> to <recep>` or `put <obj> in/on <recep>` and `o_t` is `"You move/put the <obj> to/in/on the <recep>."`.
3. `processed_objs`: Set of object instance IDs (e.g., `{"mug 2"}`) that have satisfied the required transformation:
   * Add `held_obj_{t-1}` to `processed_objs` when `o_t` matches:
     * `"You clean the <obj> using the sinkbasin..."` (for `clean`),
     * `"You heat the <obj> using the microwave..."` (for `heat`),
     * `"You cool the <obj> using the fridge..."` (for `cool`).
4. `delivered_objs` ($\mathcal{I}_{\text{deliv}}(t)$): Set of **distinct target object instance IDs** currently resting inside a `goal_recep_type` receptacle that **already satisfy all task requirements**:
   * When `held_obj_{t-1}` is successfully placed into `<recep>`:
     * If `held_obj_{t-1}` matches `target_type` AND `<recep>` matches `goal_recep_type`:
       * If $f \in \{\text{simple}, \text{pick\_two}\}$, or ($f \in \{\text{clean}, \text{heat}, \text{cool}\}$ and `held_obj_{t-1} in processed_objs`):
         add `held_obj_{t-1}` to `delivered_objs`.
   * When `<obj>` is taken *out* of `<recep>` (`You pick up the <obj> from the <recep>.`):
     * If `<obj> in delivered_objs`: remove `<obj>` from `delivered_objs` (**immediately penalizes `pick_two` same-instance recycling in `F28, F31`!**).
5. `unprocessed_in_goal`: Set of target object instance IDs placed into `goal_recep_type` **without** required processing (for $f \in \{\text{clean}, \text{heat}, \text{cool}\}$):
   * Add `held_obj_{t-1}` when an unprocessed target object is placed into `goal_recep_type` (e.g., unheated `mug 2` placed in `coffeemachine 1` in `F27`);
   * Remove `<obj>` when `<obj>` is taken back out of `goal_recep_type` (e.g., GiGPO Step 14 in `F27`).
6. `seen_target_locs`: Dict mapping each seen target object instance `<obj_id> -> <recep_id>` where it is currently known to rest (parsed from `"On/In the <recep>, you see ... a <obj> ..."` in `o_t`; removed when `<obj>` is picked up; updated if placed elsewhere).
7. `seen_lamp_locs`: Set of host receptacle IDs `<recep_id>` where a `desklamp` has been observed in `o_0, ..., o_t` (for `look_at`).
8. `lit_lamp_locs`: Set of host receptacle IDs `<recep_id>` where `use desklamp` has been executed and `o_t` confirmed `"You turn on the desklamp..."`.
9. `visited_receps_subgoal` ($U^{(p_t)}_t$): Set of receptacle IDs inspected during the **current subgoal search stage** $p_t$ (reset to `{loc_t}` whenever $p_t$ transitions to a new search stage).

---

## 3. Exact Mathematical Specification of Tier 1A & Tier 1B

### 3.1 \[T1-1\] Conflict-Aware & Instance-Locked Progress State $p_t \in \{-1, 0, \dots, K_f\}$

At any step $t$, evaluate **Terminal Success (**$p_t = K_f$**)** first, then **Conflict State (**$p_t = -1$**)**, then **Forward Progress State (**$p_t \in \{0, \dots, K_f - 1\}$**)**:

1. **Conflict / Deviation Condition (**$p_t = -1$**)**:

   $$
   \text{IsConflict}(t) \iff \Big(\text{held\_obj}_t \neq \text{None} \;\wedge\; \text{cls}(\text{held\_obj}_t) \neq \text{target\_type}\Big) \;\lor\; \Big(|\text{unprocessed\_in\_goal}(t)| > 0\Big)
   $$

   * If $\text{IsConflict}(t)$ is True, set $p_t = -1$.

2. **Forward Progress Stage (**$p_t \ge 0$ **when** $\text{IsConflict}(t)$ **is False)**:
   * **`pick_and_place_simple` (**$K_f = 2$**)**:
     * $p_t = 2$ if $|\text{delivered\_objs}(t)| \ge 1$ (or `won_t == True`);
     * $p_t = 1$ if $\text{cls}(\text{held\_obj}_t) == \text{target\_type}$;
     * $p_t = 0$ otherwise.
   * **`clean / heat / cool` (**$K_f = 3$**)**:
     * $p_t = 3$ if $|\text{delivered\_objs}(t)| \ge 1$ (or `won_t == True`);
     * $p_t = 2$ if $\text{cls}(\text{held\_obj}_t) == \text{target\_type}$ and $\text{held\_obj}_t \in \text{processed\_objs}(t)$;
     * $p_t = 1$ if $\text{cls}(\text{held\_obj}_t) == \text{target\_type}$ and $\text{held\_obj}_t \notin \text{processed\_objs}(t)$;
     * $p_t = 0$ otherwise.
   * **`look_at_obj_in_light` (**$K_f = 3$**)**:
     * $p_t = 3$ if `won_t == True` (i.e., holding target object at a host receptacle in `lit_lamp_locs`);
     * $p_t = 2$ if $\text{cls}(\text{held\_obj}_t) == \text{target\_type}$ and $\text{loc}_t \in \text{seen\_lamp\_locs}(t)$ (holding target object right at the lamp's host receptacle, ready to turn on lamp);
     * $p_t = 1$ if $\text{cls}(\text{held\_obj}_t) == \text{target\_type}$ and $\text{loc}_t \notin \text{seen\_lamp\_locs}(t)$;
     * $p_t = 0$ otherwise.
   * **`pick_two_obj_and_place` (**$K_f = 4$**)**:
     * Let $d_t = \min(2, |\text{delivered\_objs}(t)|) \in \{0, 1, 2\}$ be the number of distinct delivered target instances, and $h_t = \mathbb{I}[\text{cls}(\text{held\_obj}_t) == \text{target\_type} \;\wedge\; \text{held\_obj}_t \notin \text{delivered\_objs}(t)] \in \{0, 1\}$:
     * $p_t = 2\,d_t + h_t \in \{0, 1, 2, 3, 4\}$.

3. **Prior Potential Function** $\Phi_{\text{prior}}(f, p_t)$:

   $$
   \Phi_{\text{prior}}(f, p_t) \;=\; \begin{cases} -0.15, & \text{if } p_t = -1 \\ \dfrac{p_t}{K_f}, & \text{if } p_t \in \{0, 1, \dots, K_f\} \end{cases}
   $$

   *(Pre-lit lamp bonus in `look_at`: when* $p_t \in \{0, 1\}$ *and* $|\text{lit\_lamp\_locs}(t)| > 0$*, add* $+0.10$ *to* $\Phi_{\text{prior}}(\text{look\_at}, p_t)$ *so turning on the lamp for the first time at* $p=0 \to p=0$ *yields* $\Delta \Phi = +0.10 > 0$*).*

---

### 3.2 \[T1-2 & T1-3\] Candidate-Aware Navigation Bit $b^{\text{nav}}_t \in \{0, 1\}$ & Dynamic Sparsity Fallback

Define the active subgoal search stages as:

$$
\text{IsSearch}(f, p_t) \iff (p_t == 0) \;\lor\; (f == \text{pick\_two} \;\wedge\; p_t == 2) \;\lor\; (f == \text{look\_at} \;\wedge\; p_t == 1)
$$

For any step $t$ where $\text{IsSearch}(f, p_t)$ is True, define the **Known Candidate Receptacle Set** $\mathcal{C}_{\text{known}}(t)$ for the current subgoal:

* If $p_t == 0$: $\mathcal{C}_{\text{known}}(t) = \{\text{recep} : (\text{obj}, \text{recep}) \in \text{seen\_target\_locs}(t-1)\}$;
* If $f == \text{pick\_two}$ and $p_t == 2$: $\mathcal{C}_{\text{known}}(t) = \{\text{recep} : (\text{obj}, \text{recep}) \in \text{seen\_target\_locs}(t-1) \;\wedge\; \text{obj} \notin \text{delivered\_objs}(t)\}$;
* If $f == \text{look\_at}$ and $p_t == 1$: $\mathcal{C}_{\text{known}}(t) = \text{seen\_lamp\_locs}(t-1)$.

Define the **1-Bit Navigation Optimality Indicator** $b^{\text{nav}}_t \in \{0, 1\}$:

$$
b^{\text{nav}}_t \;=\; \begin{cases}
\underbrace{\mathbb{I}\big[\text{loc}_t \notin U^{(p_t)}_{t-1} \;\wedge\; \neg\text{IsNoOp}(t)\big]}_{\text{Epistemic Novelty } e_t \text{ (No Known Candidate)}}, & \text{if } \text{IsSearch}(f, p_t) \;\wedge\; \mathcal{C}_{\text{known}}(t) = \varnothing \\[10pt]
\underbrace{\mathbb{I}\big[\text{loc}_t \in \mathcal{C}_{\text{known}}(t) \;\wedge\; \neg\text{IsNoOp}(t)\big]}_{\text{Candidate Attraction } a_t \text{ (Known Candidate Exists!)}}, & \text{if } \text{IsSearch}(f, p_t) \;\wedge\; \mathcal{C}_{\text{known}}(t) \neq \varnothing \\[10pt]
1, & \text{if } \neg\text{IsSearch}(f, p_t)
\end{cases}
$$

* **De-Aliased Anchor Key for** $\text{AF}=0$ **Groups (`T1-2`) with Mandatory Dynamic Fallback**:
  1. **Candidate Full Key**: $\text{AnchorKey}_{\text{full}}(h_t) = (\text{norm}(o_t), p_t, b^{\text{nav}}_t)$.
  2. **Per-Iteration Sparsity Check & Fallback (Triggered on `8/150` iterations in Step 0)**:
     Compute the median anchor group size $\text{P50}_{\text{full}}$ across the current iteration's $\text{AF}=0$ groups under $\text{AnchorKey}_{\text{full}}$.
     * If $\text{P50}_{\text{full}} \ge 2$: use $\text{AnchorKey}_{\text{full}}(h_t)$;
     * If $\text{P50}_{\text{full}} < 2$: fall back to **Critical-Only De-Aliasing** for that iteration:

       $$
       \text{AnchorKey}_{\text{crit}}(h_t) \;=\; \begin{cases} (\text{norm}(o_t), p_t, b^{\text{nav}}_t), & \text{if } \text{cls}(\text{loc}_t) \in \{\text{goal\_recep\_type}, \text{tool\_recep\_type}\} \;\lor\; \text{loc}_t \in \text{seen\_lamp\_locs}(t) \\ (\text{norm}(o_t), p_t), & \text{otherwise} \end{cases}
       $$

     * Always log both the fallback trigger flag and the singleton anchor group ratio ($|G(z)| = 1$) per iteration.

---

### 3.3 \[T1-3\] Generalized No-Op Filter & Step-Level Credit for All-Failed Groups ($\text{AF}_x = 1$)

1. **State-Invariant / No-Op Action Filter** $\text{IsNoOp}(a_t, o_{t+1}, t)$:
   An action $a_t$ transitioning from $t \to t+1$ is flagged as `IsNoOp = True` iff ANY of the following holds:
   * **Environment Rejection / Text Invariance**: `o_{t+1}.strip() == "Nothing happens."` or `norm(o_{t+1}) == norm(o_t)` or `format_valid(a_t) == False`;
   * **Redundant `examine / look / inventory`**: `a_t.startswith("examine ")` or `a_t in ("look", "inventory")`;
   * **Redundant Lamp Activation (`LampAlreadyOn`)**: `a_t.startswith("use desklamp")` when `loc_t in lit_lamp_locs(t)`.

2. **Step-Level TD Credit** $\delta_t$ **in** $\text{AF}_x = 1$ **Groups (Tier 1B Mode, `enable_tier2=False`)**:
   * **Case 1 (`IsNoOp(a_t, o_{t+1}, t) == True`)**:
     Set $\lambda_t = 0$ and $\delta_t = -c_{\text{noop}}$ ($c_{\text{noop}} = 0.05$).
   * **Case 2 (`IsNoOp(a_t, o_{t+1}, t) == False`)**:
     Set $\lambda_t = \lambda_{\text{aux}} = 0.5$ and compute:

     $$
     \delta_t \;=\; \Big(\gamma\,\Phi_{\text{prior}}(f, p_{t+1}) - \Phi_{\text{prior}}(f, p_t)\Big) \;+\; \mathbb{I}\Big[p_{t+1} == p_t \;\wedge\; \text{IsSearch}(f, p_t)\Big] \cdot \kappa_{\text{nav}}\,\big(2\,b^{\text{nav}}_{t+1} - 1\big)
     $$

     where $\gamma = 0.98$ and $\kappa_{\text{nav}} = 0.05$.
   * **Backward GAE Accumulation & Batch-Std Normalized Injection**:

     $$
     A^{\text{aux}}_{j, t} \;=\; \delta_{j, t} \;+\; \gamma\,\lambda_t\,A^{\text{aux}}_{j, t+1}, \qquad A^{\text{actor}}_{j, t} \;=\; A^{\text{GiGPO}}_{j, t} \;+\; \mathbf{1}[\text{AF}_x = 1]\cdot \beta\,\frac{\tilde{A}^{\text{aux}}_{j, t}}{\hat{\sigma}_{\text{AF}} + \epsilon}
     $$

     where $\tilde{A}^{\text{aux}}_{j, t}$ is zero-mean centered within the same task group $x$ and budget bucket $b_{\text{step}}(t) \in \{0, 1, 2\}$, and $\hat{\sigma}_{\text{AF}}$ is the batch standard deviation across all $\text{AF}=1$ tokens in the iteration ($\beta = 0.5$).

---

## 4. Exact Mathematical & Engineering Specification of Tier 2 (`enable_tier2=True`)

When `enable_tier2=True`, replace the fixed heuristic potential $\Phi_{\text{prior}}$ in $\text{AF}_x = 1$ groups with the **Empirical Bayes Quotient Value Table** $\widetilde{V}(f, p_t, b^{\text{nav}}_t, b_{\text{step}}(t))$ (`T2-1`) and **Within-Group Same-Bucket Zero-Mean Centering** (`T2-2`).

### 4.1 \[T2-1a\] Exact Mapping from 180 Nominal Cartesian Cells to 96 Reachable Quotient Cells

Why a naive Cartesian product gives **180 cells**, and how the quotient mapping reduces $180 \to 144 \to \mathbf{96}$ **reachable non-terminal cells**:

1. **Full Unreduced Cartesian Product = 180 Cells**:
   Across all 6 task families, summing all defined progress states $p_t \in \{-1, 0, \dots, K_f\}$ gives:
   * `simple` ($K_f=2$): $p_t \in \{-1, 0, 1, 2\} \implies \mathbf{4}$ states;
   * `clean`, `heat`, `cool` ($K_f=3$, 3 families): $p_t \in \{-1, 0, 1, 2, 3\} \implies 5 \times 3 = \mathbf{15}$ states;
   * `look_at` ($K_f=3$): $p_t \in \{-1, 0, 1, 2, 3\} \implies \mathbf{5}$ states;
   * `pick_two` ($K_f=4$): $p_t \in \{-1, 0, 1, 2, 3, 4\} \implies \mathbf{6}$ states.
   Total $(f, p_t)$ pairs $= 4 + 15 + 5 + 6 = \mathbf{30}$ pairs. Multiplying by $b^{\text{nav}}_t \in \{0, 1\}$ ($\times 2$) and $b_{\text{step}}(t) \in \{0, 1, 2\}$ ($\times 3$) yields $30 \times 2 \times 3 = \mathbf{180}$ **Cartesian cells** (or equivalently, padding every family to 5 non-terminal states $\{-1,0,1,2,3\}$ gives $6 \times 5 \times 2 \times 3 = \mathbf{180}$ array entries).

2. **Two Structural Reductions ($180 - 36 - 48 = 96$ Reachable Non-Terminal Cells)**:
   * **Reduction 1 — Terminal Boundary Clamp ($p_t = K_f$, removes 36 cells: $180 \to 144$)**:
     When $p_t = K_f$ ($6$ families $\times 2$ nav bits $\times 3$ buckets $= \mathbf{36}$ cells), the episode has already succeeded (`won_t == True`, `done == True`). Its value is a deterministic boundary constant $\widetilde{V}_{\text{term}} \equiv 1.0$ and is never estimated from rollouts. Removing $p_t = K_f$ leaves $24$ non-terminal $(f, p_t)$ pairs ($24 \times 2 \times 3 = \mathbf{144}$ non-terminal Cartesian cells).
   * **Reduction 2 — Non-Search Navigation Bit Collapse ($\neg\text{IsSearch}(f, p_t) \implies b^{\text{nav}}_t \equiv 1$, removes 48 unreachable $b^{\text{nav}}_t=0$ cells: $144 \to 96$)**:
     By definition in §3.2, $b^{\text{nav}}_t \in \{0, 1\}$ is binary **only** in the **8 Active Search States** ($\text{IsSearch}(f, p_t) = \text{True}$), whereas in the remaining **16 Non-Search States** ($\text{IsSearch}(f, p_t) = \text{False}$), $b^{\text{nav}}_t$ is deterministically clamped to $1$ (so $(f, p_t, b^{\text{nav}}_t=0, b_{\text{step}})$ is structurally unreachable for those 16 states, removing $16 \times 1 \times 3 = \mathbf{48}$ cells).

3. **Complete Exhaustive Cell Mapping Table Across All 6 Families**:

| Family $f$ (`f_idx`) | Terminal $K_f$ (Clamped $\widetilde{V}\equiv 1.0$) | Active Search States $\text{IsSearch}=\text{True}$ ($b^{\text{nav}}_t \in \{0, 1\}$, 2 states each) | Non-Search States $\text{IsSearch}=\text{False}$ ($b^{\text{nav}}_t \equiv 1$ fixed, 1 state each) | Reachable $(p_t, b^{\text{nav}}_t)$ Pairs per Bucket | Reachable Cells ($\times 3$ Buckets) | Unreachable / Clamped out of $180$ |
|---|---|---|---|---|---|---|
| **`simple` (`0`)** | $p=2$ (6 cells) | $p=0$ $\to (0,0), (0,1)$ | $p=-1 \to (-1,1)$<br>$p=1 \to (1,1)$ | $2 + 2 = \mathbf{4}$ | $4 \times 3 = \mathbf{12}$ | $24 - 12 = 12$ |
| **`clean` (`1`)** | $p=3$ (6 cells) | $p=0$ $\to (0,0), (0,1)$ | $p=-1 \to (-1,1)$<br>$p=1 \to (1,1)$<br>$p=2 \to (2,1)$ | $2 + 3 = \mathbf{5}$ | $5 \times 3 = \mathbf{15}$ | $30 - 15 = 15$ |
| **`heat` (`2`)** | $p=3$ (6 cells) | $p=0$ $\to (0,0), (0,1)$ | $p=-1 \to (-1,1)$<br>$p=1 \to (1,1)$<br>$p=2 \to (2,1)$ | $2 + 3 = \mathbf{5}$ | $5 \times 3 = \mathbf{15}$ | $30 - 15 = 15$ |
| **`cool` (`3`)** | $p=3$ (6 cells) | $p=0$ $\to (0,0), (0,1)$ | $p=-1 \to (-1,1)$<br>$p=1 \to (1,1)$<br>$p=2 \to (2,1)$ | $2 + 3 = \mathbf{5}$ | $5 \times 3 = \mathbf{15}$ | $30 - 15 = 15$ |
| **`look_at` (`4`)** | $p=3$ (6 cells) | $p=0 \to (0,0), (0,1)$<br>$p=1 \to (1,0), (1,1)$ | $p=-1 \to (-1,1)$<br>$p=2 \to (2,1)$ | $4 + 2 = \mathbf{6}$ | $6 \times 3 = \mathbf{18}$ | $30 - 18 = 12$ |
| **`pick_two` (`5`)** | $p=4$ (6 cells) | $p=0 \to (0,0), (0,1)$<br>$p=2 \to (2,0), (2,1)$ | $p=-1 \to (-1,1)$<br>$p=1 \to (1,1)$<br>$p=3 \to (3,1)$ | $4 + 3 = \mathbf{7}$ | $7 \times 3 = \mathbf{21}$ | $36 - 21 = 15$ |
| **Total (All 6)** | **36 Terminal Cells** | **8 Search $(f,p)$ $\to$ 16 Pairs** | **16 Non-Search $(f,p)$ $\to$ 16 Pairs** | **32 Reachable Pairs** | **$32 \times 3 = \mathbf{96}$ Cells** | **$36 + 48 = \mathbf{84}$ Cells** |

4. **Canonical Array Storage & Indexing Function (Dense `(6, 5, 2, 3)` = 180 Array with 96-Cell Mask)**:
   In code, allocate a fixed numpy array of shape `(6, 5, 2, 3)` (`180` floats, where axis 1 indexes non-terminal `p_idx = p_t + 1` $\in \{0, 1, 2, 3, 4\}$ corresponding to $p_t \in \{-1, 0, 1, 2, 3\}$) plus a boolean mask `REACHABLE_MASK[6, 5, 2, 3]` with `REACHABLE_MASK.sum() == 96`:
   ```python
   def get_step_bucket(step_1idx: int) -> int:
       # step_1idx in {1, ..., 50}
       if step_1idx <= 10:
           return 0
       elif step_1idx <= 25:
           return 1
       return 2

   def canonicalize_quotient_cell(f_idx: int, p_t: int, b_nav_t: int, step_1idx: int) -> tuple[int, int, int, int]:
       """Maps any non-terminal state (p_t < K_f) to its canonical index in the (6, 5, 2, 3) array."""
       is_search = (p_t == 0) or (f_idx == 4 and p_t == 1) or (f_idx == 5 and p_t == 2)
       b_nav_canon = int(b_nav_t) if is_search else 1  # Clamp non-search states to b_nav = 1
       p_idx = p_t + 1  # Maps p_t in {-1, 0, 1, 2, 3} -> p_idx in {0, 1, 2, 3, 4}
       b_step = get_step_bucket(step_1idx)
       return (f_idx, p_idx, b_nav_canon, b_step)
   ```

---

### 4.2 \[T2-1b\] Sliding-Window First-Visit Counting & Beta-Binomial Shrinkage Formulas

At the start of training iteration $i$, freeze the completed training trajectories from the preceding $W = 4$ iterations $[i - 4, i - 1]$ (excluding validation trajectories and current iteration $i$):

1. **Donor Trajectory Pool & First-Visit Deduplication**:
   * **Donor Pool $\mathcal{D}_{[i-4, i-1]}$**: By default (`tier2_pool_mode = "mixed_only"`), include all completed training trajectories $\tau_j$ from **non-AF groups ($\text{AF}_x = 0$)** in iterations $[i - 4, i - 1]$ (with config flag `"all_groups"` available for ablation).
   * **First-Visit Deduplication per Trajectory**: For each trajectory $\tau_j \in \mathcal{D}_{[i-4, i-1]}$ with terminal binary outcome $R(\tau_j) \in \{0, 1\}$, compute the set of unique canonical non-terminal cells visited by $\tau_j$:

     $$
     \mathcal{U}(\tau_j) \;=\; \Big\{\text{canonicalize\_quotient\_cell}\big(f(\tau_j),\, p_t,\, b^{\text{nav}}_t,\, t+1\big) \;:\; 0 \le t < T_j,\; p_t < K_{f(\tau_j)}\Big\}
     $$

   * Accumulate the **First-Visit Trajectory Count** $N_{\text{hist}}(c)$ and **First-Visit Success Sum** $S_{\text{hist}}(c)$ for each of the 96 reachable cells $c = (f, p, b^{\text{nav}}, b_{\text{step}})$:

     $$
     N_{\text{hist}}(c) \;=\; \sum_{\tau_j \in \mathcal{D}_{[i-4, i-1]}} \mathbb{I}\big[c \in \mathcal{U}(\tau_j)\big], \qquad S_{\text{hist}}(c) \;=\; \sum_{\tau_j \in \mathcal{D}_{[i-4, i-1]}} \mathbb{I}\big[c \in \mathcal{U}(\tau_j)\big]\cdot R(\tau_j)
     $$

2. **Structured Prior $V_{\text{prior}}(f, p, b^{\text{nav}}, b_{\text{step}})$**:
   With step-budget decay factors $\eta(0) = 1.00,\; \eta(1) = 0.85,\; \eta(2) = 0.65$ and $\kappa_{\text{nav}} = 0.05$:

   $$
   V_{\text{prior}}\big(f, p, b^{\text{nav}}, b_{\text{step}}\big) \;=\; \begin{cases}
   -0.15, & \text{if } p = -1 \\[8pt]
   \text{clip}\!\left(\eta(b_{\text{step}}) \cdot \Big(\dfrac{p}{K_f} + \mathbb{I}[\text{IsSearch}(f, p)] \cdot \kappa_{\text{nav}}\,(2\,b^{\text{nav}} - 1)\Big),\; 0.0,\; 0.95\right), & \text{if } 0 \le p < K_f
   \end{cases}
   $$

3. **Empirical Bayes (Beta-Binomial) Posterior Shrinkage Formula**:
   With prior pseudo-count strength $\alpha_0 = 5.0$, compute the raw posterior value estimate for each reachable cell $c = (f, p, b^{\text{nav}}, b_{\text{step}})$:

   $$
   \widetilde{V}_{\text{raw}}\big(f, p, b^{\text{nav}}, b_{\text{step}}\big) \;=\; \frac{S_{\text{hist}}\big(f, p, b^{\text{nav}}, b_{\text{step}}\big) \;+\; \alpha_0 \cdot V_{\text{prior}}\big(f, p, b^{\text{nav}}, b_{\text{step}}\big)}{N_{\text{hist}}\big(f, p, b^{\text{nav}}, b_{\text{step}}\big) \;+\; \alpha_0}
   $$

---

### 4.3 \[T2-1c\] Isotonic Monotonicity Safeguard (Preventing Finite-Sample Value Inversion)

To guarantee that finite-sample noise in the 4-iteration window never inverts the credit sign of forward progress, optimal search, or conflict recovery, apply the following deterministic projection to $\widetilde{V}_{\text{raw}}$ for each family $f \in \{0..5\}$ and bucket $b \in \{0, 1, 2\}$:

1. **Intra-Stage Search Directionality Floor ($\text{IsSearch}(f, p) == \text{True}$, $\kappa_{\text{min}} = 0.02$)**:

   $$
   \widetilde{V}(f, p, 0, b) \;=\; \widetilde{V}_{\text{raw}}(f, p, 0, b), \qquad \widetilde{V}(f, p, 1, b) \;=\; \max\Big(\widetilde{V}_{\text{raw}}(f, p, 1, b),\; \widetilde{V}(f, p, 0, b) + \kappa_{\text{min}}\Big)
   $$

2. **Inter-Stage Forward Progression Floor ($p = 1, \dots, K_f - 1$, $\delta_{\text{stage}} = 0.05$)**:
   Sequentially for $p = 1, \dots, K_f - 1$:
   * If $\text{IsSearch}(f, p) == \text{False}$ (only $b^{\text{nav}}=1$ is reachable):

     $$
     \widetilde{V}(f, p, 1, b) \;=\; \max\Big(\widetilde{V}_{\text{raw}}(f, p, 1, b),\; \widetilde{V}(f, p-1, 1, b) + \delta_{\text{stage}}\Big)
     $$

   * If $\text{IsSearch}(f, p) == \text{True}$ (both $b^{\text{nav}} \in \{0, 1\}$ are reachable, i.e., `look_at` $p=1$ and `pick_two` $p=2$):

     $$
     \widetilde{V}(f, p, 0, b) \;=\; \max\Big(\widetilde{V}_{\text{raw}}(f, p, 0, b),\; \widetilde{V}(f, p-1, 1, b) + 0.02\Big)
     $$

     $$
     \widetilde{V}(f, p, 1, b) \;=\; \max\Big(\widetilde{V}_{\text{raw}}(f, p, 1, b),\; \widetilde{V}(f, p, 0, b) + \kappa_{\text{min}},\; \widetilde{V}(f, p-1, 1, b) + \delta_{\text{stage}}\Big)
     $$

3. **Conflict State Penalty Ceiling ($p = -1$, $\delta_{\text{conf}} = 0.10$)**:

   $$
   \widetilde{V}(f, -1, 1, b) \;=\; \min\Big(\widetilde{V}_{\text{raw}}(f, -1, 1, b),\; \widetilde{V}(f, 0, 0, b) - \delta_{\text{conf}},\; -0.05\Big)
   $$

---

### 4.4 \[T2-2\] Tier 2 TD Step Credit & Within-Group Same-Bucket Zero-Mean Centering

In all-failed groups ($\text{AF}_x = 1$) when `enable_tier2=True`:

1. **Step-Level TD Difference $\delta_{j, t}$**:
   * **If `IsNoOp(a_t, o_{t+1}, t) == True`**:
     $\lambda_t = 0$ and $\delta_{j, t} = -c_{\text{noop}}$ ($c_{\text{noop}} = 0.05$).
   * **If `IsNoOp(a_t, o_{t+1}, t) == False`**:
     $\lambda_t = \lambda_{\text{aux}} = 0.5$ and:

     $$
     \delta_{j, t} \;=\; \gamma\,V_{\text{next}}(t+1) \;-\; \widetilde{V}\big(\text{canonicalize\_quotient\_cell}(f, p_t, b^{\text{nav}}_t, t+1)\big) + r^{\text{lamp}}_{j, t}
     $$

     where:
     * $V_{\text{next}}(t+1) = 1.0$ if $p_{t+1} == K_f$ (`won_{t+1} == True`), else $0.0$ if $t + 1 == T_j$ (episode timeout), else $\widetilde{V}\big(\text{canonicalize\_quotient\_cell}(f, p_{t+1}, b^{\text{nav}}_{t+1}, t+2)\big)$;
     * $r^{\text{lamp}}_{j, t} = +0.10$ if $f == \text{look\_at}$ and step $t \to t+1$ turns on a `desklamp` for the first time (`LampOff -> LampOn` at $p_t \le 1$), else $0.0$.

2. **Backward GAE Accumulation**:

   $$
   A^{\text{aux}}_{j, t} \;=\; \delta_{j, t} \;+\; \gamma\,\lambda_t\,A^{\text{aux}}_{j, t+1}
   $$

3. **Within-Group Same-Budget-Bucket Zero-Mean Centering (`T2-2`)**:
   For each all-failed task group $x$ ($\text{AF}_x = 1$) and each step-budget bucket $b \in \{0, 1, 2\}$, let $S_{x, b} = \{(j, t) : \tau_j \in x,\; b_{\text{step}}(t) == b\}$ be the set of all valid steps in group $x$ belonging to bucket $b$:

   $$
   \tilde{A}^{\text{aux}}_{j, t} \;=\; \begin{cases}
   A^{\text{aux}}_{j, t} \;-\; \dfrac{1}{|S_{x, b}|}\sum_{(j', t') \in S_{x, b}} A^{\text{aux}}_{j', t'}, & \text{if } |S_{x, b}| \ge 2 \text{ and } \operatorname{Var}_{S_{x,b}}(A^{\text{aux}}) > 10^{-12} \\[8pt]
   0.0, & \text{otherwise}
   \end{cases}
   $$

4. **Batch-Std Normalized Actor Advantage Injection**:
   Let $\hat{\sigma}_{\text{AF}} = \operatorname{std}\big(\{\tilde{A}^{\text{aux}}_{j, t} : \text{AF}_x = 1\}\big)$. Inject into the actor advantage:

   $$
   A^{\text{actor}}_{j, t} \;=\; A^{\text{GiGPO}}_{j, t} \;+\; \mathbf{1}[\text{AF}_x = 1]\cdot \beta\,\frac{\tilde{A}^{\text{aux}}_{j, t}}{\hat{\sigma}_{\text{AF}} + \epsilon}, \qquad (\beta = 0.5,\; \epsilon = 10^{-6})
   $$

---

## 5. Mandatory Unit-Test Assertions on Real Traces (`test_repro_v83_rules.py`)

Codex must verify all 8 unit tests (UT-1 through UT-6 for Tier 1A/1B, and UT-7 through UT-8 for Tier 2) pass before launching training:

| Test ID | Target Trace / Module | Action & Context | Expected Behavior / Transition | Expected Credit Sign / Assertion |
|---|---|---|---|---|
| **UT-1** | `F03-RePro` Step 3 | `take pen 1 from desk 1` (goal: `pencil`) | $p_t: 0 \to -1$ (`F->D`, distractor jams hand) | **$\delta_t < 0$** (Penalized in both Tier 1B & Tier 2) |
| **UT-2** | `F03-RePro` Step 34 | `move pen 1 to desk 1` (drops distractor) | $p_t: -1 \to 0$ (`D->F`, hand freed) | **$\delta_t > 0$** (Rewarded recovery in both Tier 1B & Tier 2) |
| **UT-3** | `F27-GiGPO` Step 4 vs. Step 14 | Step 4: `move mug 2 to coffeemachine 1` (unheated)<br>Step 14: `take mug 2 from coffeemachine 1` | Step 4: $p_t: 1 \to -1$ (`F->D`)<br>Step 14: $p_t: -1 \to 1$ (`D->F`) | Step 4: **$\delta_t < 0$** (Penalized)<br>Step 14: **$\delta_t > 0$** (Rewarded recovery) |
| **UT-4** | `F12-RePro` Step 2 vs. `F01-RePro` Steps 7..48 | `F12` Step 2: 1st `use desklamp 1` (`LampOff -> On`)<br>`F01` Step 7+: repeat `use desklamp 1` (`AlreadyOn`) | `F12` Step 2: `IsNoOp=False`, `lit_lamp_locs` adds `desk 1`<br>`F01` Step 7+: `IsNoOp=True` | `F12` Step 2: **$\delta_t > 0$** (Rewarded)<br>`F01` Step 7+: **$\delta_t = -0.05 < 0, \lambda_t = 0$** |
| **UT-5** | `F12-GiGPO` Step 10 (Winning Step) | Holding `pencil 3` at `shelf 1`, executes `go to desk 1` (where `desklamp 1` is already lit) | $p_t = 1 \to p_{t+1} = 3$ (`won=True`), $\mathcal{C}_{\text{known}} = \{\text{desk 1}\}$ | **$\delta_t > 0$** (Positive terminal progress, **0 false penalty**) |
| **UT-6** | `F31-GiGPO` Step 24 vs. `F31-RePro` Step 15 | After delivering `pillow 1` ($p_t=2$, `pillow 2` seen at `armchair 1` in Step 9):<br>• GiGPO Step 24: `go to armchair 1`<br>• RePro Step 15: `take pillow 1 from sofa 1` | • GiGPO Step 24: $p_t: 2 \to 2$, $\mathcal{C}_{\text{known}}=\{\text{armchair 1}\}$, $b^{\text{nav}}_{t+1} = a_{t+1} = 1$<br>• RePro Step 15: $p_t: 2 \to 1$ (Instance recycling) | • GiGPO Step 24: **$\delta_t > 0$** (Rewarded return to candidate)<br>• RePro Step 15: **$\delta_t < 0$** (Penalized recycling) |
| **UT-7** | `QuotientValueTable` (`T2-1`) | 1. Cold-start ($N_{\text{hist}}=0$) & noisy empirical counts where $\widehat{V}_{\text{emp}}(p=1) < \widehat{V}_{\text{emp}}(p=0)$<br>2. Trajectory visiting $(f, 0, 0, 2)$ 20 times | 1. Isotonic safeguard enforces $\widetilde{V}(f, p+1, 1, b) > \widetilde{V}(f, p, 1, b)$ and $\widetilde{V}(f, -1, 1, b) \le -0.05$<br>2. First-visit deduplication increments $N_{\text{hist}}$ by **1** (not 20) | **100% of 96 reachable cells have finite support**; `D->F` transitions strictly yield $\delta_t > 0$ |
| **UT-8** | `CenterWithinBucket` (`T2-2`) | Synthetic $\text{AF}=1$ group of 8 trajectories across buckets $b \in \{0, 1, 2\}$ shifted by arbitrary constant $+c_x$ | Within each bucket $b$, $\sum_{(j,t)\in S_{x,b}} \tilde{A}^{\text{aux}}_{j,t} == 0.0$ and invariant to $+c_x$ | **Max absolute mean residual $< 10^{-12}$**; invariant to task offset $c_x$ |