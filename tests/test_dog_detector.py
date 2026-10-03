"""Decision, preprocessing, and fail-closed behavior without network downloads."""
import io
import threading
import unittest
from unittest.mock import Mock

import torch
from PIL import Image

from api.dog_detector import DogDetector, DOG_THRESHOLD, WEIGHTS
from api.inference import Classifier, DogNotFound


class DetectorTests(unittest.TestCase):
    def detector(self, labels, scores):
        detector = DogDetector.__new__(DogDetector)
        detector.device = torch.device("cpu")
        detector.dog_label = WEIGHTS.meta["categories"].index("dog")
        detector.transform = WEIGHTS.transforms()
        detector.model = Mock(return_value=[{"labels": torch.tensor(labels),
                                            "scores": torch.tensor(scores)}])
        return detector

    def test_dog_decision_ignores_confident_other_objects(self):
        dog = WEIGHTS.meta["categories"].index("dog")
        cat = WEIGHTS.meta["categories"].index("cat")
        for labels, scores, expected in [([], [], False), ([cat], [.99], False),
                ([dog], [DOG_THRESHOLD - .01], False),
                ([dog], [DOG_THRESHOLD], True), ([cat, dog], [.99, .8], True)]:
            with self.subTest(labels=labels, scores=scores):
                self.assertEqual(self.detector(labels, scores).contains_dog(
                    Image.new("RGB", (10, 10))), expected)

    def test_full_frame_is_bounded_before_tensor_allocation(self):
        detector = self.detector([], [])
        source = Image.new("RGB", (2000, 1000))
        detector.contains_dog(source)
        tensor = detector.model.call_args.args[0][0]
        self.assertEqual(tuple(tensor.shape), (3, 320, 640))
        self.assertEqual(source.size, (2000, 1000))

    def test_rejected_photo_skips_breed_model_and_releases_lock(self):
        classifier = Classifier.__new__(Classifier)
        classifier._lock = threading.Lock()
        classifier.detector = Mock()
        classifier.detector.contains_dog.return_value = False
        classifier.model = Mock()
        buffer = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, format="PNG")
        with self.assertRaises(DogNotFound):
            classifier.predict(buffer.getvalue())
        classifier.model.assert_not_called()
        self.assertFalse(classifier._lock.locked())
        classifier.detector.contains_dog.side_effect = RuntimeError("detector failed")
        with self.assertRaises(RuntimeError):
            classifier.predict(buffer.getvalue())
        classifier.model.assert_not_called()
        self.assertFalse(classifier._lock.locked())
