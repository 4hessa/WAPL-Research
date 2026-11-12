# Evidence map through report 41

Selected contrasts, not a pooled analysis. See Arabic report 42 for interpretation.

| Report | Source | Outcome | Contrast | Difference | 95% interval | Design |
|---|---|---|---|---:|---|---|
| 18 | [Persuasion](../results/transformer-pilot.json) | Spearman | weight minus baseline | -0.02529762 | [-0.08779762, 0.03125000] | diagnostic |
| 24 | [Alice](../results/horizon-replication.json) | utility MAE | horizon-matched simulation minus one-step probe | -0.00470904 | [-0.00533207, -0.00407371] | new-source diagnostic |
| 26 | [Brown](../results/utility-student.json) | utility MAE | five-feature student minus calibration | -0.00196708 | [-0.00247918, -0.00140373] | new-source diagnostic |
| 26 | [Brown](../results/utility-student.json) | utility MAE | weight features minus baseline | 0.00002605 | [-0.00004542, 0.00012159] | secondary |
| 29 | [Bryant](../results/pairwise-student.json) | top-two regret | pairwise minus pointwise regret-tuned | -0.00086343 | [-0.00146031, -0.00028801] | prospective new-source primary |
| 31 | [Edgeworth](../results/simple-control.json) | top-two regret | pairwise minus calibrated alignment | -0.00000979 | [-0.00013204, 0.00012396] | prospective new-source primary |
| 33 | [Milton](../results/sequential-policy.json) | final cross-entropy | pairwise minus alignment | 0.00004521 | [-0.00093120, 0.00119323] | equal-update primary |
| 35 | [Whitman](../results/deadline-policy.json) | final cross-entropy | pairwise minus random | 1.33273109 | [1.28081071, 1.38950331] | equal-deadline primary |
| 37 | [Whitman](../results/sparse-policy.json) | final cross-entropy | sparse pairwise minus random | 0.15021229 | [0.13496755, 0.16614090] | exposed-source exploratory |
| 39 | [Whitman](../results/loss-proxy-policy.json) | final cross-entropy | sparse highest-loss minus random | -0.00457654 | [-0.01429176, 0.00380759] | exposed-source exploratory |
| 41 | [Blake](../results/loss-replication.json) | final cross-entropy | sparse highest-loss minus random | 0.00084144 | [-0.00441628, 0.00620404] | prospective new-source primary; repeats averaged within seed |

A negative difference favors the first method for MAE, regret and cross-entropy. A positive difference favors it for Spearman. Intervals including zero do not establish equivalence. Secondary and exploratory intervals are not additional confirmatory wins.

Regenerate with `python -m tools.build_evidence_summary`. Source-result hashes are recorded in the JSON map; historical code hashes inside each source belong to its original Git revision.
