# Dog detection before breed prediction

PetSnap checks the complete decoded photo with Torchvision's
`fasterrcnn_resnet50_fpn_v2`, using the explicit `COCO_V1` weights. A dog detection
with score >= 0.5 allows the existing ResNet101 breed classifier to run. Otherwise,
the public demo returns HTTP 200 with `success: false`, `status: no_dog_detected`,
and a friendly `message`; it returns no breed predictions. Authenticated legacy
endpoints return HTTP 400 with the same message, including per-file batch errors.
Detector failures remain server errors and never bypass the check.

The detector receives an aspect-preserving whole-photo thumbnail bounded to
640 x 640 before tensor allocation. Its internal resize uses min_size=480 and
max_size=640 to limit CPU work. The breed classifier's preprocessing and private
weights are unchanged. Both stages share the existing inference lock. Detection
accepts a dog even when other objects are present; it does not crop, count dogs,
identify other animal species, or establish whether a dog is the main subject.

## Deployment

The Docker build downloads the public detector checkpoint (~167 MB) into
`TORCH_HOME=/opt/torch-cache` and verifies Torchvision's URL hash prefix. The cache
is readable by the non-root runtime user. No new Railway secrets or bucket uploads
are required. Private breed weights continue to download only at startup.

Local startup downloads the detector on first use if it is not cached. To keep the
cache in the ignored workspace directory, set TORCH_HOME to `.runtime/torch` (an
absolute path is recommended). `python -m api.dog_detector` prefetches it.

## Validation and limitations

The initial 0.5 threshold is an operating choice, not a calibrated probability.
Smoke tests preserve the golden retriever, beagle, and husky sample predictions
and reject a blank image. Additional local checks rejected scikit-image's Chelsea
cat and astronaut photos. The smaller MobileNet Faster R-CNN and SSDLite candidates
missed the husky sample at this threshold, so this version favors the larger model.
These few images are not evidence of a general accuracy rate.

Before claiming accuracy, evaluate a separate labeled set of varied breeds,
mixed breeds, puppies, distant/occluded dogs, people with dogs, cats, fish, other
animals, objects, illustrations, and stuffed toys. Measure both rejected real dogs
and accepted non-dogs; choose thresholds on validation data and report results on
held-out test data. Small dogs and unusual photos can still fail detection.

On the local Windows CPU (one Torch thread, Torch 2.4 / Torchvision 0.19), bounded
detection took roughly 6-7 seconds per image in the five-image smoke test. Railway's
pinned CPU runtime and hardware differ: check actual latency and peak memory after
deployment. Keep one replica/worker and watch memory during uploads, not just idle.

Run `python -m unittest discover -s tests -p test_dog_detector.py -v` for isolated
decision/preprocessing tests and `python -m unittest discover -s tests -p test_demo.py -v`
for real-checkpoint API tests. The latter needs private breed weights and the
public detector cache. Public quota tests are in `tests/test_public_demo.py`.

Model reference: https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html
Local negative fixtures (not shipped with the app):
https://github.com/scikit-image/scikit-image/tree/v0.19.3/skimage/data
