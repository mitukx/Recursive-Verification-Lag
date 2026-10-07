# Retained free-T4 evidence

Read [the outcome and limits](../../../docs/free_colab_pilot_evidence.md).

- `raw/`: untouched second Colab export, including both failed diagnostic attempts.
- `source_snapshots/`: exact four diagnostic source/config files identified by its manifest.
- `scorecard.json`: derived report, reproducible with `python -m scripts.verify_colab_evidence_bundle`.

Original first export SHA256:
`4c04989bde5db1a1318999b35f9cc9fd7f46e98b64e26ea91ab7d9a92a2a7732`.
Second export (`rvl-pilot-evidence-20261007T162508345719Z-diagnostic-02.zip`) SHA256:
`f7c633ac9fdc575b2eb77aec94e8ab704e67bc979f79c0d8ff3f355ec1de0f9d`.
Archive hashes are transport metadata; per-file raw hashes are in `raw/bundle-manifest.json`.
The ZIP itself is delivered separately and is not required by CI.

Pilot research source: `30b3a438ac7bd77a0cc216ff250873a9b8818925`.
Pilot systems source: `1a63903d22e1233594d1f1f0c269f7f55baf99b7`.
Diagnostic 1: `0306aba603c60fe06e090bc477400b3db4c0eb31` (source/copy checksum error).
Diagnostic 2: `e7595d07bcec572c8459dfc9b1365ca194fe4627` (corrected parity passes; historical replay gate fails).

Task text/answers are retained from the pinned [GSM8K dataset](https://github.com/openai/grade-school-math)
under its MIT license; see `GSM8K_LICENSE`. No model weights are included.
Hashes do not provide remote hardware attestation. The observed negative outcome
and failed replay tolerance must accompany any portfolio claim.
