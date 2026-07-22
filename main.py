"""
AI Closet backend — receives a clothing photo, removes the background,
tags it via a vision model, and returns structured item data.

Run locally with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import json
import os
import random
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import anthropic

app = FastAPI(title="AI Closet API")


@app.on_event("startup")
async def log_tagging_mode():
    if USE_MOCK_TAGGING:
        print("⚠️  No ANTHROPIC_API_KEY found — using MOCK tagging (fake random tags).")
    else:
        print("✅ ANTHROPIC_API_KEY found — using real Claude vision tagging.")

# --- Storage setup (local disk for now; swap for S3/Supabase storage later) ---
STORAGE_DIR = Path("storage/items")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

# If no API key is set, the server automatically falls back to mock tagging
# instead of calling Claude. This lets you build and test the whole app
# end-to-end for free, then flip over to real tagging once you have a key.
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
USE_MOCK_TAGGING = API_KEY is None

client = anthropic.Anthropic(api_key=API_KEY) if not USE_MOCK_TAGGING else None


class ClosetItem(BaseModel):
    id: str
    imageUrl: str
    category: str
    color: str
    pattern: str | None
    formality: str
    season: list[str]


TAGGING_PROMPT = """You are looking at a photo of a single clothing item with
its background removed. Respond with ONLY a JSON object, no other text, in
this exact shape:

{
  "category": "top | bottom | dress | outerwear | shoes | accessory",
  "color": "primary color, one or two words",
  "pattern": "solid | striped | plaid | floral | graphic | null",
  "formality": "casual | smart-casual | formal | athletic",
  "season": ["subset of: spring, summer, fall, winter"]
}
"""


@app.post("/items/upload", response_model=ClosetItem)
async def upload_item(photo: UploadFile = File(...)):
    if photo.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Photo must be JPEG or PNG")

    raw_bytes = await photo.read()

    # 1. Remove background so the item is isolated on a clean canvas
    try:
        cutout_bytes = remove(raw_bytes)
    except Exception as e:
        raise HTTPException(500, f"Background removal failed: {e}")

    # 2. Save the processed image to disk with a unique id
    item_id = str(uuid.uuid4())
    filename = f"{item_id}.png"
    filepath = STORAGE_DIR / filename
    Image.open(io.BytesIO(cutout_bytes)).save(filepath)

    # 3. Tag the item — mock tags if no API key, real vision model otherwise
    if USE_MOCK_TAGGING:
        tags = mock_tag_clothing_item()
    else:
        tags = tag_clothing_item(cutout_bytes)

    return ClosetItem(
        id=item_id,
        imageUrl=f"/storage/items/{filename}",
        category=tags["category"],
        color=tags["color"],
        pattern=tags.get("pattern"),
        formality=tags["formality"],
        season=tags["season"],
    )


def mock_tag_clothing_item() -> dict:
    """
    Returns plausible-looking fake tags so you can build and test the
    full camera -> upload -> tag -> display pipeline without an API key.
    Swap to real tagging automatically once ANTHROPIC_API_KEY is set.
    """
    categories = ["top", "bottom", "dress", "outerwear", "shoes", "accessory"]
    colors = ["navy", "black", "white", "olive", "burgundy", "beige", "denim blue"]
    patterns = [None, "solid", "striped", "plaid", "floral", "graphic"]
    formalities = ["casual", "smart-casual", "formal", "athletic"]
    seasons_options = [["spring", "summer"], ["fall", "winter"], ["spring", "summer", "fall", "winter"]]

    return {
        "category": random.choice(categories),
        "color": random.choice(colors),
        "pattern": random.choice(patterns),
        "formality": random.choice(formalities),
        "season": random.choice(seasons_options),
    }


def tag_clothing_item(image_bytes: bytes) -> dict:
    import base64

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": TAGGING_PROMPT},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if the model added them despite instructions
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Vision model returned unparseable tags: {text}")
