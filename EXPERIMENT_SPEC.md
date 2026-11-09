# Experiment specification

Recorded before model evaluation on 9 November 2026.

- Task: classify CLINC150 in-scope utterances into 150 intents while studying label-selection policies; separately evaluate a confidence-based out-of-scope rejection heuristic.
- Dataset: pinned original `data_full.json`; retain official validation/test rows; remove normalized training texts also present in validation/test, including out-of-scope evaluation text. Preserve class labels and official evaluation splits.
- Representation: frozen, pinned `all-MiniLM-L6-v2`, masked mean pooling, normalized 384-dimensional vectors, 256 wordpiece maximum. No encoder training or fine-tuning.
- Classifier: scikit-learn multinomial logistic regression, C=10, L-BFGS, max_iter=500, tolerance=0.0001. Same head configuration for every policy and budget.
- Policies: uniform random acquisition, maximum predictive entropy, and margin sampling with round-robin balancing across predicted classes. The latter uses predicted classes, not hidden ground-truth classes.
- Seeds: 42, 43, 44, 45, 46. Within a seed, all policies share an initial class-balanced labeled set of three examples per intent (450 labels). This starter set uses an oracle in the simulation and is an idealized assumption, not a realistic free annotation step.
- Training-label budgets: 450, 900, 1500, 3000. Acquisition is batched between checkpoints. Budget counts exclude the separately available 3000 in-scope validation labels and any out-of-scope evaluation labels. No actual human annotation time or monetary savings are measured.
- Selection functions receive unlabeled-pool predictions and IDs, never hidden labels. Oracle labels become accessible only after a batch is selected. Classifier fitting uses only acquired training indices.
- Choose the deployment policy by mean validation macro-F1 at the final 3000-label budget across the five seeds. Deployment seed is fixed at 42; it is not selected by validation or test score.
- Freeze all policy trajectories and model-selection choices before calculating test learning curves. Test curves are descriptive reports, not feedback for selection.
- Primary metrics: known-intent accuracy and macro-F1. Compare policies at equal label budgets; report all seeds and mean/standard deviation. Report paired policy-minus-random macro-F1 differences at 3000 labels, with a 95% Student-t interval across seeds. This interval reflects seed variability on this fixed benchmark, not uncertainty over new datasets.
- Full-data reference: same classifier on the entire cleaned training pool, reported separately from the label-budget comparison.
- Rejection heuristic: choose the fifth percentile of known validation maximum predicted probabilities (lower quantile) as a gate for the selected seed-42 model. No out-of-scope examples are used to train the head or choose the gate. Evaluate known acceptance, out-of-scope rejection, rejection AUROC, and accepted known-intent accuracy. This is not a formal conformal or out-of-distribution coverage guarantee.
- Report local latency on 50 validation utterances, including frozen encoding and head scoring; exclude startup, model loading, and HTTP overhead. Provide a local prediction API and correctness checks.
- No state-of-the-art, production-scale, independent-authorship, guaranteed hiring, novel algorithm, or actual annotation-cost reduction claim.
