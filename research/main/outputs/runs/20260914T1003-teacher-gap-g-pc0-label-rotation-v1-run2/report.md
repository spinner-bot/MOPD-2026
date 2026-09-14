# G-PC0 label-rotation diagnostic result

Created 2026-09-14 10:14 UTC+8 by Codex (CINECA Leonardo), after Slurm job `57639554` completed the 1046 frozen diagnostic.

All 256/256 sample-condition records passed the completeness and provenance gate. After mapping each displayed label back to the frozen semantic option and averaging its raw log-likelihood across four cyclic rotations, S/T-Q/T-QA/Generic-T scored 3/3/9/2 of 16. T-QA produced six S-wrong to T-QA-right changes and zero regressions, so the only preregistered decision is `LABEL_INTERFACE_MASKED_G_SIGNAL`.

Posthoc description shows that raw displayed outputs remained strongly label-biased: A accounted for 55/64 S and 43/64 T-QA generations. Semantic aggregation removed enough of that bias for the correct-option margin to move from mean −0.402 under S to +0.176 under T-QA. Gains occurred in directed path (1), sequential transform (3), and grid trace (2); set relation had none. These family counts are descriptive and cannot be used to select natural data.

The result establishes that the present T-QA representation carries behaviorally usable signal for these procedural tasks when the answer interface is neutralized. It does not replace G-PC0 v1's frozen failure, establish an effect on natural images, authorize training data, or by itself unlock natural G0. The authoritative interpretation is progress report 1047.
