from groq import Groq
from iflens.schema.conditions import ConditionSchema, ImageConditions

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt

import base64
import json

# qwen/qwen3.6-27b no longer exists on Groq (NotFoundError, 2026-11-13); pin the id and record it.
VISION_MODEL = "qwen/qwen3.8-27b"

MAX_SIDE_FOR_CLASSIFICATION = 1024  # weather/lighting does not need full resolution; big files got BadRequest

CLASSIFICATION_BASE_PROMPT = "Given the schema below, provide a JSON classifying each attribute of the image in the format {attribute: name}, where the name is not None, according to the available names:\n"

def __get_condition_classification_prompt(condition_schema: ConditionSchema) -> str:
    return CLASSIFICATION_BASE_PROMPT + condition_schema.to_markdown()

def encode_image(image_path, max_side: int = MAX_SIDE_FOR_CLASSIFICATION):
    """Base64 JPEG of the image, downscaled so its longer side is <= max_side."""
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError(f"Cannot encode image: {image_path}")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

@retry(stop=stop_after_attempt(3), reraise=True)  # reraise: keep the real API error in the record
def classify_image(image_path: str,
                   condition_schema: ConditionSchema,
                   client: Groq,
                   model: str = VISION_MODEL,
                   temperature: float = 0,
                   seed: int | None = None) -> ImageConditions:
    base64_image = encode_image(image_path=image_path)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": __get_condition_classification_prompt(condition_schema=condition_schema)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    }
                ]
            }
        ],
        temperature=temperature,
        seed=seed,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
        stop=None,
    )
    
    obj = completion.choices[0].message.content
    
    assert obj is not None, "No JSON provided in the response!"
    
    return ImageConditions.from_dict(schema=condition_schema, data=json.loads(obj))

if __name__ == "__main__":
    
    load_dotenv()
    client = Groq()
    condition_schema = ConditionSchema.from_yaml(path="conditions-example.yaml")
    
    image_path = r"examples\street\example.png"
    
    conditions = classify_image(image_path=image_path,
                                condition_schema=condition_schema,
                                client=client)
    
    print(conditions.to_prompt())
    
    print("-- Variation below --")
    
    print(conditions.get_variation().to_prompt())
    