# Local line-ending variants of manual-review manifests

During GitHub synchronization, two untracked local copies occupied the paths used by the sample manifests from the server home repository. Their raw SHA-256 values differed because the local worktree used CRLF while the committed blobs use LF. `git hash-object --path` normalized each archived file to the exact committed Git blob, confirming content equivalence before the no-fast-forward merge. The copies were moved unchanged here; the complete canonical manifests were then restored at their original paths from Git.

- `manual-review-sample-v1-local-crlf.json`: 22,871 local bytes; raw SHA-256 `b549dadf4c91ffaa45f34f7ba9311cc82b97d2995a48876bda484835b8e7388c`; normalized Git blob `3af4e4cea712e2efb8d75327869ee05952e58759`.
- `manual-review-sample-v2-local-crlf.json`: 22,459 local bytes; raw SHA-256 `c12fc754e7e7be3017672635840f7016446f8e4e5d3c4ef930a072491abbe331`; normalized Git blob `4e160578620cbea58432f2f7056ee3fc76e89b0d`.

These are redundant line-ending variants retained for provenance. The canonical manifests are `audits/manual-review-sample-v1.json` and `audits/manual-review-sample-v2.json`; v2 is the corrected selection used by the review packet.
