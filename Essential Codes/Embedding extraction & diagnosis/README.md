# Embedding extraction and integrity diagnosis

[Research overview](../../README.md)

These utilities establish the input representation and audit possible subject-block anomalies before formal geometry experiments.

| Script | Role |
|---|---|
| `extract_cogbci_biot_embeddings_5s.py` | Extract BIOT representations from five-second CogBCI windows. |
| `diagnose_subject_block_streaming_v1.py` | Audit HDF5 key alignment, subject norms/variance, effective rank, duplicates, centroid similarity, and identity probes. |

The diagnostic random projection does not replace the formal experiment's SVD scan. Its suspect-subject lists are audit inputs rather than independent findings that particular participants are defective.

```bash
python3 extract_cogbci_biot_embeddings_5s.py --help
python3 diagnose_subject_block_streaming_v1.py --help
```

Embedding extraction requires the original model/data environment. No separate extraction or integrity-audit result report is included in the three source PDFs.
