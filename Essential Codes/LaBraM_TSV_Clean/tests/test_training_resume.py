import tempfile
import unittest
from pathlib import Path

from labram_tsv.io import dump_json, sha256_file, write_csv
from labram_tsv.training import _run_is_complete


class TrainingResumeTests(unittest.TestCase):
    def test_resume_requires_identity_protocol_w0_epochs_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "adapted_checkpoint.pth"
            delta = root / "delta_weights.npz"
            checkpoint.write_bytes(b"checkpoint")
            delta.write_bytes(b"delta")
            checkpoint_hash = sha256_file(checkpoint)
            delta_hash = sha256_file(delta)
            write_csv(
                root / "epoch_metrics.csv",
                [{"epoch": epoch, "balanced_accuracy": 0.5} for epoch in range(1, 21)],
            )
            artifacts = {
                "delta_weights_sha256": delta_hash,
                "adapted_checkpoint_sha256": checkpoint_hash,
            }
            dump_json(
                root / "metadata.json",
                {
                    "task": "Rest",
                    "replicate": 1,
                    "epochs": 20,
                    "protocol_fingerprint": "protocol",
                    "base_full_backbone_digest": "w0",
                    "artifacts": artifacts,
                },
            )
            dump_json(
                root / "_SUCCESS.json",
                {
                    "phase": "final",
                    "task": "Rest",
                    "replicate": 1,
                    "protocol_fingerprint": "protocol",
                    "base_full_backbone_digest": "w0",
                    **artifacts,
                },
            )
            self.assertTrue(
                _run_is_complete(root, "Rest", 1, "protocol", "w0")
            )
            self.assertFalse(
                _run_is_complete(root, "Rest", 1, "different", "w0")
            )
            delta.write_bytes(b"tampered")
            self.assertFalse(
                _run_is_complete(root, "Rest", 1, "protocol", "w0")
            )


if __name__ == "__main__":
    unittest.main()
