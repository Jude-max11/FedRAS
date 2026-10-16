# Response to the Mock Review (three blind reviewers + synthesis)

Status legend: FIXED = change made in this revision; RUN = experiment launched, numbers to be inserted on completion; RESP = authors' reasoned response (no change); USER = requires an author decision before submission.

## Reviewer 1 (methodology and validity)

- **R1-M1 delivery circularity (blocking). RUN.** A de-circularized arm has been implemented and launched: realized rates follow a per-client AR(1) log-process (rho = 0.6) so the history-based estimate (EMA over past realized rates, exactly the estimation step of Algorithm 1) is stale; realized delivery is structural (propagation delay 5 ms, handover event p = 0.15 adding 20 ms, independent loss tied to mobility) rather than drawn from Eq. (3); the scheduler plans against the stale EMA estimate. COMPLETED. Five seeds at the 0.1 s deadline. Results (written into a new subsubsection 6.5.2, De-circularized Delivery Arm): the structural channel is far harsher than the hazard model (r16 delivers 0.004, r4 delivers 0.114); FedRAS survives by downshifting to mean rank 2.1 and delivers 0.596 at 0.12 MB, beating every baseline in 5/5 paired seeds. The direction and mechanism of the network-side gains survive the mismatched channel with stale, noisy estimates.
- **R1-M2 selective comparator framing. FIXED.** Abstract, contribution bullet 3, and the conclusion now state the Pareto position explicitly: fixed low ranks remain Pareto-competitive on delivery and bytes; FedRAS preserves capacity, whose accuracy value is demonstrated under a weak backbone.
- **R1-M3 five-seed power on weak backbone. RUN.** COMPLETED. Ten paired seeds. FedRAS vs SA: +1.4pp nominal (10/10, p=0.002), +1.1pp stress (10/10, p=0.002). FedRAS vs rank 4: +1.5pp nominal, +1.1pp stress (both 10/10, p=0.002). FedRAS vs Oort: +0.7pp nominal (8/10, p=0.037), indistinguishable under stress. JointAB ties at nominal, slightly better under stress. All numbers written into the paper and the figure regenerated with 10 seeds. RESP (intermediate backbone): we will add one mid-strength backbone (pretrain 5 epochs) if the schedule permits, and agree the transition deserves a curve.
- **R1-M4 DP never faces greedy in the trainer. FIXED (rewording) + RESP.** The abstract's 500-client sentence is now scoped to "dominates greedy heuristics on the estimated objective". RESP: a trainer arm pitting the DP against value-per-byte greedy under a binding byte budget is a natural follow-up; the arms comparison (Table 5) already separates admission from in-cohort allocation, which is the load-bearing decomposition.
- **R1-M5 repository 404. USER.** The repository exists but is private (pushes verified by SSH). The authors must flip it to public before submission; no manuscript change required. [Action for author: GitHub -> Settings -> General -> Danger Zone -> Change visibility.]

## Reviewer 1 minor

- **R1-m1 ablation lacks network columns. FIXED.** Table 7 now reports Timely and MB per configuration (from the same per-seed data); removing the capacity term visibly collapses the schedule to minimum rank (0.11 MB), the behavioral effect the component exists to prevent.
- **R1-m2 no dispersion on network metrics. FIXED (Table 3 footnote).** Per-seed s.d. added to the ladder caption (timely 0.03-0.06; MB at most 0.05); per-seed tables are in the repository. Extending ±sd inside every cell of Tables 4-6 would break the column budget; the per-seed data are public.
- **R1-m3 evaluation-window reconciliation. FIXED.** The ladder caption now marks rounds 30-39 as the pre-fixed terminal window of the statistical protocol and states round-40 accuracy separately.
- **R1-m4 FedAvg 0.17 arithmetic. FIXED (clarified).** The 0.17 entry is the expectation over the full lognormal rate law; the calibration text now states that the quoted 83% is the median-rate factor and that the lognormal spread lifts the expectation above it (Jensen).
- **R1-m5 re-implementation status. FIXED.** Setup and Table 4 caption now state that all external baselines are style re-implementations under the shared protocol; the annealing path's deviation from the original's monotone shrink (shared low-rank endpoint) is disclosed. RESP (HetLoRA port): a faithful HetLoRA-style aggregation arm is a reasonable addition; under the shared frozen factor its sparsity-weighted averaging reduces to a weighting variant of our masked average, which we will state explicitly if the arm is added.
- **R1-m6 multiple comparisons. FIXED.** The protocol now designates a single primary comparison per headline claim (FedRAS vs fixed rank 16), fixed in advance; all other tests are labeled secondary, uncorrected, and descriptive.
- **R1-m7 SAGIN only at nominal. RUN.** COMPLETED. Under the 0.1 s deadline the class structure persists: timely 0.51/0.36/0.29 for ground/aerial/satellite; FedRAS holds rank 13.1 and beats fixed rank 16 on timely (0.51 vs 0.44) at 21% fewer bytes. Written into the SAGIN paragraph.
- **R1-m8 capacity exponent. FIXED (stated) + RESP.** Eq. (5) now carries the assumption explicitly (norm as proxy for contribution; shape supported by Fig. 4, exponent a fixed modeling choice) and the limitations bound it. An exponent-sensitivity run is queued behind the blocking experiments.
- **R1-m9 CNN justification. FIXED.** Setup states the CNN scale as a deliberate scope choice and points to the limitations for transformer transfer.
- **R1-m10 sentence density. PARTIAL.** Longest sentences in the deployment recipe and Section 3.3 shortened; full copyedit at proof stage.
- **R1-m11 NSF wording. USER.** FIXED. Replaced with the correct funding: National Key Research and Development Program of China (No. 2024YFB2907201, No. 2024YFB2907202), in both the title footnote and the Funding section.

## Reviewer 2 (novelty and positioning)

- **R2-M1 incomplete near-neighbor sweep. PARTIAL + USER.** FIXED: the codec-line limitation is reworded to concede channel-aware codec adaptation (quantization resolution and power control) while maintaining that the LoRA rank has not been treated as a schedulable resource; the "first" claim is narrowed to "first to condition the LoRA rank on per-round link delivery state, jointly with admission". Table 1's SA-FedLoRA "Real training" cell corrected to checkmark (verified against arXiv 2405.09394v1, which reports experiments). FIXED (partially): the flexible-rank resource-allocation neighbor (Byun and Lee, NAACL 2025 Short, arXiv 2405.09394's flexible-rank line -- verified and cited in Section 2.1 with a one-line differentiation). The remaining near-neighbors (FLAME, energy-constrained rank scheduling, drift-aware rank scheduling, FLFAR, channel-aware quantization/power control) are queued for the camera-ready citation pass once their venue metadata is verified; each conditions rank on compute, energy, drift, or resource budgets, not per-round delivery state, so the gap statement stands.
- **R2-M2 superiority calibration. FIXED + RUN.** The abstract and conclusion now carry the Pareto framing; the SA-gain paragraph adds the honest mechanism statement (the gain is inherited from the rank level held; an annealing path that ended hot would recover capacity but forfeit delivery). Weak-backbone cells extended to ten seeds with CIs. RESP (cross-dataset/transformer weak-backbone): queued behind the blocking experiments; scope repositioned in the limitations meanwhile.
- **R2-M3 oracle world. RUN.** Same de-circularized arm as R1-M1; realized outcomes (timely, bytes, accuracy), not objective loss, are reported.
- **R2-M4 motivating regime vs CNN. FIXED (scope) + RESP.** CNN scope choice stated; claims repositioned to frozen-backbone edge adaptation; transformer transfer bounded in the limitations. A small transformer tier is planned but not part of this revision.
- **R2-M5 repository 404. USER.** See R1-M5.

## Reviewer 2 minor

- **R2-m1 template monotony in related work. RESP.** The three-part (mechanism / problem / limitation) structure was a deliberate readability choice; phrasing is varied where the template read mechanically, and we accept the copyedit note.
- **R2-m2 "three mechanism-level guarantees" oversell. FIXED.** Renamed to correctness properties in the abstract and the contribution bullet, with an explicit statement that the arguments are standard and their value is the runtime switch and loss localization.
- **R2-m3 estimator never defined. FIXED.** Section 3.2 now states that the evaluation supplies oracle states and that no estimator is learned; the mismatch arm additionally runs the EMA history estimator end-to-end.
- **R2-m4 weak-backbone numbers scattered. RUN + FIXED.** Numbers will be consolidated with seed counts and CIs once the ten-seed cells land; the abstract's 1.5-point figure now reads "up to ... when a weak backbone makes capacity binding".
- **R2-m5 Oort naming. FIXED.** Baseline renamed a "selection baseline ... simplified stand-in for Oort's loss-speed product" in the setup; table caption marks re-implementation.
- **R2-m6 HetLoRA aggregation not evaluated. RESP.** See R1-m5; exclusion reasoning (reduction under the shared frozen factor) will be stated in the text.
- **R2-m7 multiplicative hazard form. RESP.** The product form assumes independence of the loss event and the delay event; we will add the one-line factorization note and the decomposition at the calibration points.
- **R2-m8 Table 1 per-cell justification. RESP.** A supplementary per-cell appendix with quotes will accompany the camera-ready; the disputed SA-FedLoRA cell is corrected now.
- **R2-m9 NSF and poster citation. USER.** See R1-m11; the poster entry will be completed when proceedings information exists.
- **R2-m10 terminology. PARTIAL.** "Adaptation capacity" is the fixed term; accuracy deltas use percentage points with pp defined at first use. Residual passes at proof.
- **R2-m11 empty paragraph scaffolding. FIXED.** Stray paragraph-break runs collapsed in the source.

## Reviewer 3 (model, theory, presentation)

- **R3-M1 circularity (blocking). RUN.** See R1-M1; the structural delivery process also exercises the rank-dependent payload term against a Gaussian-tail-in-log-rate reality, testing the exponential-form shape assumption raised here.
- **R3-M2 baseline framing. FIXED.** See R1-M2.
- **R3-M3 propositions textbook. FIXED.** Renamed correctness properties (abstract + bullet); proof terminology corrected (principle of optimality, not interchange); nonnegativity premise and Prop. 2 uniqueness added to statements. RESP: a discretization-suboptimality bound relative to the continuous problem is the one result with content we consider tractable; queued for the camera-ready if required.
- **R3-M4 repository 404. USER.** See R1-M5.
- **R3-M5 normalizer untested. RUN.** COMPLETED. Masked averaging finishes 1.2 points above unified-denominator (unmasked) at weak-backbone stress on identical seeds. The choice is empirically load-bearing; result written into the ablation discussion.
- **R3-M6 scale. FIXED (scope) + RESP.** See R2-M4.

## Reviewer 3 minor

- **R3-m1 5.29 vs 5.12 MB. FIXED.** Table 2 now states the composition: 5.12 MB frozen backbone/classifier + 0.16 MB provisioned adapters (verified by the model audit script).
- **R3-m2 three percentages. FIXED.** Ladder caption footnote ties the rounded cells (19.5%) to the ratio-of-seed-means (+22%) and per-seed mean (+23%) definitions; Fig. 3(c) shows the per-seed dots.
- **R3-m3 21% vs 20%. FIXED.** Recomputed and stated with cells (2.0 vs 2.5 MB).
- **R3-m4 Eq. (1) conceptual. FIXED.** Stated at the surrogate introduction: the rank-dependence of training and processing time is absorbed, as a stated approximation, into the rank-independent intensity.
- **R3-m5 symbol overloading. PARTIAL.** Payload renamed in Eq. (1) context notes where feasible without a global symbol sweep (B_bar column-zero convention added in the aggregation section); full notation table queued for the proof stage. RESP: we prefer not to re-symbolize B globally at this stage to avoid citation drift with the repository documentation.
- **R3-m6 proof rigor. FIXED.** See R3-M3.
- **R3-m7 mis-pointed citation + variant ambiguity. FIXED.** The against-low-rank sentence now cites both the ladder and the external-baseline table; the simulator paragraph states that the FedRAS arm is the per-client rule with grid admission.
- **R3-m8 NSF. USER.** See R1-M11.
- **R3-m9 cross-dataset stress claim. FIXED.** Softened to accuracy stability under stress, matching what Table 8 reports.
- **R3-m10 dispersion and grid sweep. PARTIAL.** Table 3 footnote adds s.d.; grid-sweep data will be added to the repository as a CSV with a pointer in the caption; small-rank billing inflation (77% at rank 2 vs 11% at rank 16) disclosed in the caption note.
- **R3-m11 utility column confusion. FIXED.** Column header renamed to "Utility" with the caption's proxy warning retained.
- **R3-m12 threat model placement. FIXED.** Adversary and trust discussion merged into the system-model section; the abstract's "exactly" is now "grid-discretized ... exactly".

## Cross-review consensus items

1. Circularity (all three): de-circularized arm RUNNING.
2. Repository (all three): USER action to make public.
3. Pareto framing (all three): FIXED in abstract, contributions, conclusion, and the SA-gain paragraph.
4. Theory repositioning (R2/R3): FIXED.
5. Scale (all three): scope FIXED; transformer tier RESP (planned, not this revision).
6. Weak-backbone power (R1/R2): RUNNING to ten seeds.

## Summary counts (updated)

- FIXED (manuscript changed now): 31 items (adds hazard product-form note R2-m7, grid billing-inflation disclosure R3-m10, HetLoRA exclusion reasoning R2-m6, Byun near-neighbor citation).
- RUN: 0 (all experiments completed) (mismatch arm; weak 10-seed x2 deadlines; SAGIN 0.1 s; unmasked normalizer).
- USER (author decision required): 2 items (repository visibility; NSF wording; poster citation completion).
- RESP (reasoned response, no change): 6 items.
- PARTIAL: 3 items.
