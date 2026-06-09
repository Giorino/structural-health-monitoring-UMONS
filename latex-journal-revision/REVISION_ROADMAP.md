# Revision Roadmap

## Overview

- Source manuscript: rejected Scientific Reports draft in `latex/`
- Working copy: `latex-journal-revision/`
- Goal: rebuild the paper into a journal-defendable manuscript using the current strict CNN + physics evidence from the repo
- Strategy: narrow claims first, then rebuild methods/results around the defendable protocol, then revise title/abstract/discussion

## P1: Must Fix

| ID | Comment Summary | Reviewer | Type | Section | Suggested Action |
|---|---|---|---|---|---|
| P1-1 | The paper overclaims crack localization without documented physical crack position ground truth. | R1, R2 | Major | Title, Abstract, Introduction, Discussion | Reframe the paper as crack-associated event detection or damage-state detection unless independent crack-location evidence can be added. |
| P1-2 | The old evaluation uses too few independent runs and over-relies on pooled window metrics. | R1 | Major | Methods, Results | Replace the old 29-run / 5-test-run framing with strict run-level validation and report experiment-level metrics first. |
| P1-3 | The label-generation protocol is unclear and may be circular. | R1, R2 | Major | Methods | Add a precise crack-labeling subsection describing source of labels, synchronization path, and limitations. If independent ground truth is absent, state that explicitly and narrow the claim. |
| P1-4 | The method is presented as raw-signal localization, but the actual representation is heavily engineered and context-dependent. | R1 | Major | Abstract, Introduction, Methods | Rewrite the method description honestly: simple CNN on peak-aligned multiplexed FBG windows with synchronized loading/physics covariates and a physics-consistency filter. |
| P1-5 | The mechanics-informed claims are stronger than the current physical validation supports. | R1 | Major | Methods, Discussion | Downgrade mechanics language from strong physics-informed inference to physics-motivated descriptors / filtering unless uncertainty analysis is added. |
| P1-6 | The old headline metrics are not the current defendable ones. | Internal evidence | Major | Abstract, Results, Conclusion | Replace old window-level `F1=0.819` claims with the current strict reproducible baseline: experiment precision `0.8333`, recall `0.6667`, F1 `0.7407`. |

## P2: Should Fix

| ID | Comment Summary | Reviewer | Type | Section | Suggested Action |
|---|---|---|---|---|---|
| P2-1 | Add learning curves and overfitting evidence. | R2 | Minor | Results / Supplement | Add learning-curve figure or refer to saved training curves where available. |
| P2-2 | Add simpler baselines to justify the CNN. | R1, R2 | Minor | Results | Include at least one simple non-deep baseline and the simple strict CNN baseline as the main comparator. |
| P2-3 | Clarify that recall is still limited for SHM deployment. | Internal evidence | Minor | Discussion | Add an SHM limitations paragraph explaining that the current recall is not yet deployment-grade and that threshold tuning alone cannot fix it. |
| P2-4 | Document the merge strategy and why the chosen one was retained. | Internal evidence | Minor | Methods | State that raw reconstruction with legacy boundary hints gave better aligned and more reproducible results than pure raw peak search. |

## P3: Consider

| ID | Comment Summary | Reviewer | Type | Section | Suggested Action |
|---|---|---|---|---|---|
| P3-1 | Add a limitations-oriented paragraph about specimen count and lack of external validation. | R1 | Editorial | Discussion | Add an explicit limitations subsection. |
| P3-2 | Add a statement on unmatched raw files and data completeness. | Internal evidence | Editorial | Data / Methods | Briefly note unmatched raw runs if they remain excluded from supervised evaluation. |

## Cross-Reviewer Pattern

Both reviewers challenged the same core issue from different angles: the paper currently claims more than the evidence supports. The revision must therefore be a scope correction first, not a cosmetic rewrite.

## Suggested Revision Order

1. Rewrite the claim scope:
   title, abstract, and introduction
2. Rebuild the Methods section:
   dataset, labels, merge/alignment, strict validation, current CNN + physics protocol
3. Rebuild Results around current defendable metrics:
   experiment-level first, window-level second
4. Rewrite Discussion and limitations:
   sample size, label provenance, recall limits, non-localization framing
5. Only after those are stable:
   figures, polishing, formatting, and target-journal adaptation

## Progress Notes

- Completed:
  title, abstract, introduction, and the main claim-scope correction toward crack-event detection
- Completed:
  Results and Discussion were rewritten around the current strict run-level operating point (`precision = 0.8333`, `recall = 0.6667`, `F1 = 0.7407`)
- Completed:
  Methods were rewritten so that the representation, architecture, and training protocol now match the retained nine-feature sequence CNN baseline
- Completed:
  New manuscript figures were generated for the retained protocol, including a selected-pipeline schematic and a strict LOOCV threshold/results summary
- Completed:
  the manuscript was migrated into the MDPI article template as a self-contained submission folder with copied figures and bibliography
- Remaining polish:
  external compilation and final journal-specific formatting still depend on the local LaTeX toolchain
