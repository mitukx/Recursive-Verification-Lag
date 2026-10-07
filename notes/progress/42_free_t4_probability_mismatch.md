# 42. Free T4 pilot and fresh-policy numerical mismatch

2026-10-08: the original locked Qwen/GSM8K pilot completed all six seed/arm
conditions on a free Tesla T4. Mean accuracy fell from 31.25% to 18.75% in both
trusted and shuffled arms. No capability improvement is established.

Fresh behavior scores had a large same-weight mismatch: the sampler inherited
Qwen's repetition penalty while the learner used raw logits. Neutral generation
controls and temperature-aware learning were implemented and tested. A separate
zero-update T4 diagnostic measured maximum corrected log-ratio 7.4625e-5, below
its declared 1e-4 tolerance, with zero clipping on six short samples.

The diagnostic nevertheless failed overall: reconstructing old scores with the
penalty left error 0.00110242, above the separate 1e-4 tolerance. The tolerance
was retained. The earlier checksum-verifier failure was also archived. All raw
predictions, training histories, tokens/scores, environment records and failures
are retained, with a CI-recomputed scorecard.

See [full measurements, provenance and limits](../../docs/free_colab_pilot_evidence.md).
A corrected learning replication and an independent task split remain open.
