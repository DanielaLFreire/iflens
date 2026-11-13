import argparse
import base64
import os
import shutil
from datetime import UTC, datetime
from random import Random

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.interactions import Interaction
from groq import Groq

from iflens.functions.classification import VISION_MODEL, classify_image
from iflens.functions.generation import BASE_GENERATION_PROMPT
from iflens.reproducibility import (
    ASPECT_TOLERANCE,
    aspect_deviation,
    derive_seed,
    sample_names,
    sha256_bytes,
    sha256_file,
    write_record,
)
from iflens.schema.conditions import ConditionSchema
from iflens.utils import time_signed_print as print


def check_env_keys():
    keys_to_check = [
        ("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY")),
        ("GROQ_API_KEY", os.getenv("GROQ_API_KEY")),
    ]

    for name, value in keys_to_check:
        if not value or value == "YOUR_KEY_HERE":
            raise ValueError(f"No {name} found inside .env!")


def get_file_ext(filename: str) -> str:
    return os.path.splitext(filename)[1]


def is_image(filename: str) -> bool:
    SUPPORTED_IMAGE_FORMATS: list[str] = [".png", ".jpeg", ".jpg"]

    ext = get_file_ext(filename=filename)
    return ext.lower() in SUPPORTED_IMAGE_FORMATS


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of encoded image bytes, or None if undecodable."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else (int(img.shape[1]), int(img.shape[0]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iflens: condition-based generative augmentation")
    p.add_argument("--dataset", default=os.path.join("examples", "cars"))
    p.add_argument("--conditions", default="conditions-example.yaml")
    p.add_argument("--n", type=int, default=3, help="images to sample (<= dataset size)")
    p.add_argument("--seed", type=int, required=True, help="master seed (recorded in every output)")
    p.add_argument("--keep", nargs="*", default=["planet"],
                   help="condition classes that must not be varied")
    p.add_argument("--model", default="gemini-3.1-flash-lite-image",
                   help="generation model id; pin an exact id and record it")
    p.add_argument("--vision-model", default=VISION_MODEL)
    p.add_argument("--output", default="output")
    return p.parse_args()


def main():
    args = parse_args()

    load_dotenv()
    check_env_keys()

    print("STEP 0 --- SETUP")
    os.makedirs("input", exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    print("STEP 1 --- SAMPLING IMAGES TO AUGMENT (seeded)")
    images = [f for f in os.listdir(args.dataset) if is_image(f)]
    print(f"- {len(images)} images found")
    sampled = sample_names(images, args.n, args.seed)
    print(f"- Sampled {len(sampled)} images.")

    print("STEP 2 --- Creating folder with sampled images and annotations")
    experiment_time: str = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
    experiment_folder: str = os.path.join("input", experiment_time)
    os.makedirs(experiment_folder, exist_ok=True)

    for sample_img in sampled:
        print(f"- Copying image {sample_img}")
        try:
            img_path = os.path.join(args.dataset, sample_img)
            annotation = os.path.splitext(img_path)[0] + ".txt"
            shutil.copy(img_path, os.path.join(experiment_folder, sample_img))
            shutil.copy(annotation, os.path.join(experiment_folder, os.path.basename(annotation)))
        except Exception as e:
            print(f"- Error: {e!r}")
            print("Skipping...")

    print("- STEP 3 --- Generating Augmentations")
    client = genai.Client()
    groq_client = Groq()

    images = sorted(f for f in os.listdir(experiment_folder) if is_image(f))
    output_folder: str = os.path.join(args.output, experiment_time)
    os.makedirs(output_folder, exist_ok=True)

    schema = ConditionSchema.from_yaml(path=args.conditions)
    conditions_file_sha256 = sha256_file(args.conditions)
    print(f"- Ready to augment {len(images)} images")

    for i, image in enumerate(images):
        tag = f"({i + 1}/{len(images)})"
        stem = os.path.splitext(image)[0]
        record: dict = {
            "image": image,
            "master_seed": args.seed,
            "conditions_file": args.conditions,
            "conditions_file_sha256": conditions_file_sha256,
            "keep_classes": args.keep,
            "generation_model_requested": args.model,
            "vision_model": args.vision_model,
            "status": "started",
        }
        try:
            print(tag, f"- Image {image}")
            in_filename = os.path.join(experiment_folder, image)
            out_filename = os.path.join(output_folder, f"AUG_{stem}.png")

            with open(in_filename, "rb") as f:
                image_bytes = f.read()
            record["input_sha256"] = sha256_bytes(image_bytes)
            record["input_size_wh"] = image_size(image_bytes)

            classify_seed = derive_seed(args.seed, image, "classify")
            variation_seed = derive_seed(args.seed, image, "variation")
            record["classify_seed"] = classify_seed
            record["variation_seed"] = variation_seed

            conditions = classify_image(
                in_filename, schema, groq_client,
                model=args.vision_model, temperature=0, seed=classify_seed,
            )
            variation = conditions.get_variation(keep=args.keep, rng=Random(variation_seed))
            record["original_conditions"] = conditions.to_dict()
            record["variation_conditions"] = variation.to_dict()

            prompt = BASE_GENERATION_PROMPT + variation.to_prompt()
            record["generation_prompt"] = prompt
            print(tag, f"- Original: {conditions.to_dict()}  ->  Variation: {variation.to_dict()}")

            record["generation_started_utc"] = datetime.now(tz=UTC).isoformat()
            interaction = client.interactions.create(
                model=args.model,
                input=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                        "mime_type": "image/png",
                    },
                ],
            )
            assert isinstance(interaction, Interaction)
            record["response_id"] = getattr(interaction, "id", None)
            record["generation_model_returned"] = getattr(interaction, "model", None)
            assert interaction.output_image and interaction.output_image.data, (
                "Generated image is None!"
            )

            out_bytes = base64.b64decode(interaction.output_image.data)
            with open(out_filename, "wb") as f:
                f.write(out_bytes)
            record["output_file"] = os.path.basename(out_filename)
            record["output_sha256"] = sha256_bytes(out_bytes)
            record["output_size_wh"] = image_size(out_bytes)
            # Geometry check only; it does NOT prove the objects were preserved. The output
            # resolution differs from the input (e.g. height 768), so compare the aspect ratio:
            # normalised (YOLO) boxes stay valid only if the ratio is kept.
            dev = aspect_deviation(record["input_size_wh"], record["output_size_wh"])
            record["aspect_ratio_deviation"] = dev
            record["same_aspect_ratio"] = dev is not None and dev <= ASPECT_TOLERANCE
            record["status"] = "ok"
            print(tag, "- Done!")
        except Exception as e:
            record["status"] = "error"
            record["error"] = repr(e)
            print(tag, f"- Error: {e!r}")
            print(tag, "Skipping...")
        finally:
            write_record(os.path.join(output_folder, f"AUG_{stem}.json"), record)


if __name__ == "__main__":
    main()
