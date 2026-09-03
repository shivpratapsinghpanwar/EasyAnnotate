"""Offline augmentation, with the annotations carried along.

Augmentation is where a small dataset becomes a usable one, and where a careless
implementation quietly ruins it: flip the pixels but not the boxes and every
object is now mirrored onto empty background, which is worse than no
augmentation at all. Here the geometry goes through the same
:func:`~data_factory.core.imaging.transform_annotation` every other operation
uses, so a flip cannot desynchronise.

Two decisions worth stating:

*Offline, not on-the-fly.* Modern trainers augment per epoch in the data loader,
which is strictly better for the model. Offline augmentation is still worth
having, because it is inspectable -- you can open the folder and look at what
the model will actually be trained on, which is how you catch a hue shift that
turns every red light green.

*Seeded.* The same seed and the same inputs produce the same output files,
byte for byte. An augmented dataset you cannot reproduce is not a dataset, it
is an accident.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np

from .. import imaging, parallel
from ..annotations.model import Annotation
from ..registry import Param, operation
from ..types import Context, OpResult
from ..writer import DatasetWriter
from .image import OUTPUT_FORMATS, SOURCE_FORMATS, _load

AUGMENT_PARAMS = (
    Param("folder", "dir", "Dataset to augment.", required=True, must_exist=True),
    Param("output", "dir", "Where the augmented dataset is written.", required=True),
    Param(
        "multiplier",
        "int",
        "Augmented variants to generate per source image.",
        default=3,
        minimum=1,
        maximum=50,
    ),
    Param("include_original", "bool", "Copy the untouched original in as well.", default=True),
    Param("seed", "int", "Random seed; the same seed reproduces the same output.", default=0),
    # -- geometric
    Param("flip_horizontal", "bool", "Mirror left-right.", default=True),
    Param("flip_vertical", "bool", "Mirror top-bottom.", default=False),
    Param("rotate90", "bool", "Rotate by a random quarter turn.", default=False),
    Param(
        "rotate",
        "float",
        "Maximum rotation in degrees (0 disables).",
        default=0.0,
        minimum=0.0,
        maximum=180.0,
    ),
    Param(
        "crop",
        "float",
        "Maximum random zoom-in, as a fraction of the frame (0 disables).",
        default=0.0,
        minimum=0.0,
        maximum=0.5,
    ),
    # -- photometric
    Param(
        "brightness",
        "float",
        "Maximum brightness shift as a fraction (0 disables).",
        default=0.2,
        minimum=0.0,
        maximum=1.0,
    ),
    Param(
        "contrast",
        "float",
        "Maximum contrast shift as a fraction (0 disables).",
        default=0.2,
        minimum=0.0,
        maximum=1.0,
    ),
    Param(
        "saturation",
        "float",
        "Maximum saturation shift as a fraction (0 disables).",
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    ),
    Param(
        "hue",
        "float",
        "Maximum hue shift in degrees (0 disables).",
        default=0.0,
        minimum=0.0,
        maximum=180.0,
    ),
    Param("blur", "float", "Maximum blur radius in pixels (0 disables).", default=0.0, minimum=0.0),
    Param(
        "noise",
        "float",
        "Maximum gaussian noise as a fraction (0 disables).",
        default=0.0,
        minimum=0.0,
        maximum=0.5,
    ),
    Param(
        "grayscale",
        "float",
        "Probability a variant is converted to greyscale.",
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    ),
    Param("cutout", "int", "Number of random erased boxes (0 disables).", default=0, minimum=0),
    Param(
        "min_visibility",
        "float",
        "Fraction of an object that must remain visible for it to be kept.",
        default=0.3,
        minimum=0.0,
        maximum=1.0,
    ),
    Param("source_format", "choice", "Input format.", default="auto", choices=SOURCE_FORMATS),
    Param("format", "choice", "Output format.", default="same", choices=OUTPUT_FORMATS),
    Param("quality", "int", "JPEG/WebP quality.", default=95, minimum=1, maximum=100),
    Param("recursive", "bool", "Include sub-folders.", default=True),
    Param(
        "splits",
        "strs",
        "Only augment these splits (blank means all). Usually: train",
        default=None,
    ),
    Param("workers", "int", parallel.WORKERS_HELP, default=0, minimum=0, maximum=32),
)


@operation(
    "augment.dataset",
    title="Augment dataset",
    category="augment",
    summary="Generate seeded, annotation-aware variants: flips, rotation, colour, blur, noise, cutout.",
    params=AUGMENT_PARAMS,
    destructive=True,
    notes=(
        "Each variant independently rolls each enabled augmentation, so a run\n"
        "with flip, rotate and brightness on gives a genuine mixture rather\n"
        "than every image receiving all three.\n"
        "\n"
        "Augment the training split only. Augmenting validation makes the score\n"
        "measure your augmentation rather than your model, and it usually\n"
        "flatters it. Pass --splits train on a split dataset.\n"
        "\n"
        "Pick augmentations that match how the model will be used, not whatever\n"
        "sounds thorough. Vertical flip on a door dataset teaches the model\n"
        "that upside-down doors are normal; they are not.\n"
        "\n"
        "The same --seed reproduces the run exactly."
    ),
)
def augment_dataset(params: dict[str, Any], ctx: Context) -> OpResult:
    result = OpResult(op="augment.dataset", dry_run=ctx.dry_run)
    dataset, target_format = _load(params, ctx, result)
    if dataset is None:
        return result

    wanted = set(params["splits"] or [])
    plan = _Plan(params)
    if not plan.enabled:
        result.fail(
            "no augmentations are enabled -- everything is at its zero default. "
            "Turn some on, e.g. --brightness 0.25 --rotate 10"
        )
        return result

    writer = DatasetWriter(
        params["output"], target_format, ctx, classes=dataset.labels, quality=params["quality"]
    )
    multiplier = params["multiplier"]
    applied: Counter[str] = Counter()
    generated = 0
    copied = 0
    dropped = 0
    total = len(dataset.items)

    quality = params["quality"]
    seed = params["seed"]

    def build_variants(item):
        """Decode once, generate every variant, hand back encoded bytes.

        Pure and thread-safe: each variant makes its own generator from a
        stable seed, so nothing here depends on execution order.
        """
        if item.image is None:
            return None
        if wanted and item.split and item.split not in wanted:
            return ("passthrough", [])
        array = imaging.load(item.image)
        produced = []
        for variant in range(multiplier):
            rng = np.random.default_rng(_seed_for(seed, item.stem, variant))
            pixels, annotation, used = plan.apply(array, item.annotation, rng)
            produced.append(
                (annotation, imaging.encode(pixels, item.image.suffix, quality=quality), used)
            )
        return ("augment", produced)

    def note_failure(item, exc: Exception) -> None:
        result.warnings.append(f"{item.image.name if item.image else item.stem}: {exc}")

    for item, outcome in parallel.map_bounded(
        dataset.items,
        build_variants,
        ctx=ctx,
        workers=params["workers"],
        label=lambda item: item.stem,
        on_error=note_failure,
    ):
        if outcome is None:
            continue
        kind, produced = outcome

        if kind == "passthrough":
            # Out of scope: pass it through untouched so the output is still a
            # complete dataset rather than a train-only fragment.
            writer.add(
                item.annotation,
                stem=item.stem,
                source=item.image,
                split=item.split,
                suffix=item.image.suffix,
            )
            copied += 1
            continue

        if params["include_original"]:
            writer.add(
                item.annotation,
                stem=item.stem,
                source=item.image,
                split=item.split,
                suffix=item.image.suffix,
            )
            copied += 1

        for variant, (annotation, encoded, used) in enumerate(produced, start=1):
            dropped += max(0, len(item.annotation.shapes) - len(annotation.shapes))
            applied.update(used)
            writer.add(
                annotation,
                stem=f"{item.stem}_aug{variant}",
                encoded=encoded,
                source=item.image,
                split=item.split,
                suffix=item.image.suffix,
            )
            generated += 1

    report = writer.close()
    result.actions.extend(writer.actions)
    result.counts.update(
        {
            "source_images": total,
            "generated": generated,
            "originals_kept": copied,
            "total_images": generated + copied,
            "objects_dropped": dropped,
        }
    )
    result.warnings.extend(dataset.warnings)
    result.warnings.extend(report["warnings"])
    result.data.update(
        {
            "output": params["output"],
            "format": target_format,
            "seed": params["seed"],
            "augmentations_enabled": sorted(plan.enabled),
            "times_applied": dict(sorted(applied.items())),
        }
    )
    if dropped:
        result.warnings.append(
            f"{dropped} object(s) rotated or cropped out of frame and were dropped; "
            f"lower --min-visibility to keep partial objects"
        )
    return result


def _seed_for(seed: int, stem: str, variant: int) -> int:
    """A stable per-(image, variant) seed.

    Deliberately not ``hash((seed, stem, variant))``: Python randomises string
    hashing per process, so that would reproduce within one run and silently
    differ between runs -- the exact opposite of the guarantee this operation
    makes. blake2b is stable across processes, machines and versions.

    Seeding per image rather than per run also means adding one image to a
    dataset does not reshuffle every other image's augmentations.
    """
    key = f"{seed}\x00{stem}\x00{variant}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**32)


class _Plan:
    """The enabled augmentations, and how to roll one variant from them."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.min_visibility = params["min_visibility"]
        self.geometric = [
            name
            for name, on in (
                ("flip_horizontal", params["flip_horizontal"]),
                ("flip_vertical", params["flip_vertical"]),
                ("rotate90", params["rotate90"]),
                ("rotate", params["rotate"] > 0),
                ("crop", params["crop"] > 0),
            )
            if on
        ]
        self.photometric = [
            name
            for name, on in (
                ("brightness", params["brightness"] > 0),
                ("contrast", params["contrast"] > 0),
                ("saturation", params["saturation"] > 0),
                ("hue", params["hue"] > 0),
                ("blur", params["blur"] > 0),
                ("noise", params["noise"] > 0),
                ("grayscale", params["grayscale"] > 0),
                ("cutout", params["cutout"] > 0),
            )
            if on
        ]

    @property
    def enabled(self) -> list[str]:
        return self.geometric + self.photometric

    def apply(
        self, array: np.ndarray, annotation: Annotation, rng: np.random.Generator
    ) -> tuple[np.ndarray, Annotation, list[str]]:
        used: list[str] = []
        pixels = array
        height, width = pixels.shape[:2]
        steps: list[imaging.Transform] = []

        for name in self.geometric:
            # Half the time, per augmentation -- so variants differ from each
            # other instead of all receiving the identical stack.
            if rng.random() < 0.5:
                continue
            transform = self._geometric(name, width, height, rng)
            if transform is None:
                continue
            steps.append(transform)
            width, height = transform.out_size
            used.append(name)

        if steps:
            combined = imaging.chain(*steps) if len(steps) > 1 else steps[0]
            pixels = combined.image(pixels)
            annotation = imaging.transform_annotation(
                annotation, combined, min_visibility=self.min_visibility
            )
        else:
            annotation = Annotation(
                filename=annotation.filename,
                width=width,
                height=height,
                depth=annotation.depth,
                shapes=list(annotation.shapes),
                folder=annotation.folder,
                source_path=annotation.source_path,
            )

        for name in self.photometric:
            if rng.random() < 0.5:
                continue
            pixels = self._photometric(name, pixels, rng)
            used.append(name)

        return pixels, annotation, used

    def _geometric(
        self, name: str, width: int, height: int, rng: np.random.Generator
    ) -> imaging.Transform | None:
        if name == "flip_horizontal":
            return imaging.FlipH(out_size=(width, height))
        if name == "flip_vertical":
            return imaging.FlipV(out_size=(width, height))
        if name == "rotate90":
            turns = int(rng.integers(1, 4))
            out = (height, width) if turns % 2 else (width, height)
            return imaging.Rot90(out_size=out, turns=turns, src_size=(width, height))
        if name == "rotate":
            limit = float(self.params["rotate"])
            degrees = float(rng.uniform(-limit, limit))
            return imaging.make_rotate(width, height, degrees)
        if name == "crop":
            fraction = float(rng.uniform(0.0, self.params["crop"]))
            new_w = max(8, int(width * (1.0 - fraction)))
            new_h = max(8, int(height * (1.0 - fraction)))
            left = int(rng.integers(0, max(1, width - new_w + 1)))
            top = int(rng.integers(0, max(1, height - new_h + 1)))
            return imaging.Crop(out_size=(new_w, new_h), left=left, top=top)
        return None

    def _photometric(
        self, name: str, pixels: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        p = self.params
        if name == "brightness":
            return imaging.adjust_brightness(pixels, 1.0 + float(rng.uniform(-p["brightness"], p["brightness"])))
        if name == "contrast":
            return imaging.adjust_contrast(pixels, 1.0 + float(rng.uniform(-p["contrast"], p["contrast"])))
        if name == "saturation":
            return imaging.adjust_saturation(pixels, 1.0 + float(rng.uniform(-p["saturation"], p["saturation"])))
        if name == "hue":
            return imaging.adjust_hue(pixels, float(rng.uniform(-p["hue"], p["hue"])))
        if name == "blur":
            return imaging.blur(pixels, float(rng.uniform(0.5, max(0.5, p["blur"]))))
        if name == "noise":
            return imaging.add_noise(pixels, float(rng.uniform(0.0, p["noise"])), rng)
        if name == "grayscale":
            return imaging.to_grayscale(pixels) if rng.random() < p["grayscale"] else pixels
        if name == "cutout":
            return imaging.cutout(pixels, int(p["cutout"]), 0.15, rng)
        return pixels
