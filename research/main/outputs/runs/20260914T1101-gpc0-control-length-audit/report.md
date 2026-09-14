# G-PC0 control-length audit

Created 2026-09-14 11:01 UTC+8 by Codex (CINECA Leonardo), triggered by the control-variable audit recorded in 1055.

This CPU-only, input-only audit applies the frozen Qwen3.5-4B tokenizer to the 16 immutable G-PC0 base inputs. T-Q/T-QA/Generic-T token lengths have medians 49/74/49, means 50.5/82.125/49, and ranges 46–58/68–110/49–49. T-QA is therefore 1.5102 times the Generic-T median length.

The result shows that the historical Generic-T matched T-Q but did not match T-QA. It does not read performance records and does not alter the archived G-PC0 decision. It motivated the result-blind natural G0 v0.4 change to separate Generic-Q and Generic-QA. No natural claim or GPU authorization follows.
