import tempfile
import unittest
from pathlib import Path

from src.rvl_systems.weight_sync import WeightManifest, WeightPublisher


class WeightSyncTest(unittest.TestCase):
    def test_publish_verify_and_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = WeightPublisher(tmp)
            manifest = publisher.publish_bytes(b"model-state-v1")
            self.assertEqual(manifest.version, 1)
            self.assertTrue((Path(tmp) / manifest.artifact).exists())
            publisher.verify(manifest)
            self.assertFalse(publisher.all_acknowledged(["w0", "w1"]))
            publisher.acknowledge("w0", 1)
            self.assertEqual(publisher.workers_behind(["w0", "w1"]), {"w1": 1})
            publisher.acknowledge("w1", 1)
            self.assertTrue(publisher.all_acknowledged(["w0", "w1"]))

    def test_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = WeightPublisher(tmp)
            manifest = publisher.publish_bytes(b"good")
            (Path(tmp) / manifest.artifact).write_bytes(b"bad")
            with self.assertRaises(RuntimeError):
                publisher.verify(manifest)

    def test_ack_cannot_move_backwards_or_ahead(self):
        with tempfile.TemporaryDirectory() as tmp:
            publisher = WeightPublisher(tmp)
            publisher.publish_bytes(b"v1")
            publisher.publish_bytes(b"v2")
            publisher.acknowledge("w0", 2)
            with self.assertRaises(ValueError):
                publisher.acknowledge("w0", 1)
            with self.assertRaises(ValueError):
                publisher.acknowledge("w1", 3)


if __name__ == "__main__":
    unittest.main()
