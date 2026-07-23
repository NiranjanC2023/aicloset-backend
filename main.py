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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import anthropic
from google import genai
from google.genai import types

app = FastAPI(title="AI Closet API")


@app.on_event("startup")
async def log_tagging_mode():
    print(f"Tagging mode: {TAGGING_PROVIDER}")

# --- Storage setup (local disk for now; swap for S3/Supabase storage later) ---
STORAGE_DIR = Path("storage/items")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

# Tagging provider is chosen automatically based on which API key is set:
# ANTHROPIC_API_KEY -> Claude, GEMINI_API_KEY -> Gemini (free tier),
# neither -> mock tags. This lets you develop for free and swap providers
# without touching code, just environment variables.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if ANTHROPIC_API_KEY:
    TAGGING_PROVIDER = "claude"
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
elif GEMINI_API_KEY:
    TAGGING_PROVIDER = "gemini"
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    TAGGING_PROVIDER = "mock"

# --- Simple persisted item list (a JSON file for now; swap for a real DB later) ---
ITEMS_FILE = Path("storage/items.json")


def load_items() -> list[dict]:
    if not ITEMS_FILE.exists():
        return []
    return json.loads(ITEMS_FILE.read_text())


def save_items(items: list[dict]) -> None:
    ITEMS_FILE.write_text(json.dumps(items, indent=2))


class ClosetItem(BaseModel):
    id: str
    imageUrl: str
    category: str
    color: str
    pattern: Optional[str]
    formality: str
    season: list[str]
    thickness: Optional[str] = None   # "light" | "medium" | "heavy"
    fit: Optional[str] = None         # e.g. "skinny", "baggy", "cropped", "oversized"
    lastSuggestedAt: Optional[str] = None  # ISO timestamp, used for repeat-avoidance


TAGGING_PROMPT = """You are looking at a photo of a single clothing item with
its background removed. Respond with ONLY a JSON object, no other text, in
this exact shape:

{
  "category": "top | bottom | dress | outerwear | shoes | accessory",
  "color": "primary color, one or two words",
  "pattern": "solid | striped | plaid | floral | graphic | null",
  "formality": "casual | smart-casual | formal | athletic",
  "thickness": "light | medium | heavy",
  "fit": "see guidance below based on category, or null if not applicable",
  "season": ["subset of: spring, summer, fall, winter"]
}

Guidance:
- thickness: judge the fabric weight from how it drapes/looks (e.g. a thin
  cotton tee is "light", a hoodie or knit sweater is "medium", a wool coat
  or heavy denim jacket is "heavy"). Use thickness plus material to decide
  season — light items skew spring/summer, heavy items skew fall/winter,
  medium can span multiple seasons.
- fit: only applies to bottoms, tops, and outerwear.
  - For bottoms (pants/jeans/shorts): one of "skinny", "straight", "baggy",
    "bootcut", "relaxed", "wide-leg".
  - For tops/hoodies/sweaters: one of "cropped", "regular", "oversized",
    "baggy", "fitted".
  - For shoes/accessories/dresses: use null.
"""


@app.post("/items/upload", response_model=ClosetItem)
async def upload_item(
    photo: UploadFile = File(...),
    formalityHint: Optional[str] = Form(None),
):
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

    # 3. Tag the item using whichever provider is configured
    if TAGGING_PROVIDER == "claude":
        tags = tag_with_claude(cutout_bytes, formalityHint)
    elif TAGGING_PROVIDER == "gemini":
        tags = tag_with_gemini(cutout_bytes, formalityHint)
    else:
        tags = mock_tag_clothing_item()

    new_item = ClosetItem(
        id=item_id,
        imageUrl=f"/storage/items/{filename}",
        category=tags["category"],
        color=tags["color"],
        pattern=tags.get("pattern"),
        formality=tags["formality"],
        season=tags["season"],
        thickness=tags.get("thickness"),
        fit=tags.get("fit"),
    )

    items = load_items()
    items.append(new_item.model_dump())
    save_items(items)

    return new_item


@app.get("/items", response_model=list[ClosetItem])
async def list_items():
    return load_items()


class OutfitRequest(BaseModel):
    occasion: Optional[str] = None  # e.g. "casual", "work", "date night"
    formalityPreference: Optional[str] = None  # user's own definition of "formal"
    avoidRepeatDays: Optional[int] = None  # skip items suggested within N days


class OutfitSuggestion(BaseModel):
    itemIds: list[str]
    reasoning: str


class OutfitResponse(BaseModel):
    outfits: list[OutfitSuggestion]


OUTFIT_PROMPT = """You are a fashion stylist. Here is a JSON list of clothing
items in someone's closet, each with an id, category, color, pattern,
formality, fit, thickness, and season:

{items_json}

Occasion: {occasion}
{formality_context}

Suggest up to 3 complete outfits using ONLY the item ids provided above.
Each outfit should make sense together (matching formality, complementary
colors, compatible fits — e.g. don't force baggy items with fitted ones
unless it's clearly intentional streetwear layering — and appropriate for
the occasion). Include a top+bottom OR a dress, optionally with
outerwear/shoes/accessories if suitable items exist. Don't invent items
that aren't in the list.

Respond with ONLY a JSON object, no other text, in this exact shape:

{{
  "outfits": [
    {{"itemIds": ["id1", "id2"], "reasoning": "one sentence on why this works"}}
  ]
}}
"""


@app.post("/outfits/generate", response_model=OutfitResponse)
async def generate_outfits(request: OutfitRequest):
    all_items = load_items()

    if len(all_items) < 2:
        raise HTTPException(
            400, "Add at least 2 items to your closet before generating outfits."
        )

    # Repeat-avoidance: exclude items suggested within the last N days,
    # but fall back to the full closet if that would leave too few items
    # to build an outfit from (small closets shouldn't get stuck empty).
    candidate_items = all_items
    if request.avoidRepeatDays and request.avoidRepeatDays > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=request.avoidRepeatDays)
        fresh_items = [
            i for i in all_items
            if not i.get("lastSuggestedAt")
            or datetime.fromisoformat(i["lastSuggestedAt"]) < cutoff
        ]
        if len(fresh_items) >= 2:
            candidate_items = fresh_items

    if TAGGING_PROVIDER == "claude":
        outfits = generate_outfits_with_claude(
            candidate_items, request.occasion, request.formalityPreference
        )
    elif TAGGING_PROVIDER == "gemini":
        outfits = generate_outfits_with_gemini(
            candidate_items, request.occasion, request.formalityPreference
        )
    else:
        outfits = mock_generate_outfits(candidate_items)

    # Mark suggested items so future requests can avoid repeating them.
    suggested_ids = {item_id for o in outfits for item_id in o.itemIds}
    if suggested_ids:
        now_iso = datetime.now(timezone.utc).isoformat()
        for item in all_items:
            if item["id"] in suggested_ids:
                item["lastSuggestedAt"] = now_iso
        save_items(all_items)

    return OutfitResponse(outfits=outfits)


def _build_outfit_prompt(
    items: list[dict], occasion: Optional[str], formality_preference: Optional[str]
) -> str:
    # Only send the fields the model actually needs — keeps the prompt small
    # and avoids leaking image URLs into the text prompt.
    slim_items = [
        {
            "id": i["id"],
            "category": i["category"],
            "color": i["color"],
            "pattern": i.get("pattern"),
            "formality": i["formality"],
            "fit": i.get("fit"),
            "thickness": i.get("thickness"),
            "season": i["season"],
        }
        for i in items
    ]
    formality_context = (
        f"The user describes their idea of 'formal' as: {formality_preference}. Use this to interpret formality levels."
        if formality_preference
        else ""
    )
    return OUTFIT_PROMPT.format(
        items_json=json.dumps(slim_items),
        occasion=occasion or "any occasion",
        formality_context=formality_context,
    )


def _parse_outfit_response(text: str) -> list[OutfitSuggestion]:
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        return [OutfitSuggestion(**o) for o in data["outfits"]]
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(500, f"Couldn't parse outfit suggestions: {e}")


def generate_outfits_with_claude(
    items: list[dict], occasion: Optional[str], formality_preference: Optional[str] = None
) -> list[OutfitSuggestion]:
    prompt = _build_outfit_prompt(items, occasion, formality_preference)
    response = claude_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_outfit_response(response.content[0].text)


def generate_outfits_with_gemini(
    items: list[dict], occasion: Optional[str], formality_preference: Optional[str] = None
) -> list[OutfitSuggestion]:
    prompt = _build_outfit_prompt(items, occasion, formality_preference)
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[prompt],
    )
    return _parse_outfit_response(response.text)


def mock_generate_outfits(items: list[dict]) -> list[OutfitSuggestion]:
    """
    Simple rule-based pairing so outfit generation works even without an
    API key: pair each dress alone, or match tops with bottoms of similar
    formality. Not smart, but proves the feature end-to-end for free.
    """
    tops = [i for i in items if i["category"] == "top"]
    bottoms = [i for i in items if i["category"] == "bottom"]
    dresses = [i for i in items if i["category"] == "dress"]

    outfits: list[OutfitSuggestion] = []

    for dress in dresses[:2]:
        outfits.append(
            OutfitSuggestion(
                itemIds=[dress["id"]],
                reasoning=f"A simple {dress['color']} {dress['formality']} dress works as a complete outfit on its own.",
            )
        )

    for top in tops:
        same_formality_bottoms = [b for b in bottoms if b["formality"] == top["formality"]]
        if same_formality_bottoms:
            bottom = random.choice(same_formality_bottoms)
            outfits.append(
                OutfitSuggestion(
                    itemIds=[top["id"], bottom["id"]],
                    reasoning=f"Pairs a {top['color']} {top['category']} with a {bottom['color']} {bottom['category']} — both {top['formality']}, so they match in vibe.",
                )
            )
        if len(outfits) >= 3:
            break

    return outfits[:3]


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
    thicknesses = ["light", "medium", "heavy"]
    bottom_fits = ["skinny", "straight", "baggy", "bootcut", "relaxed", "wide-leg"]
    top_fits = ["cropped", "regular", "oversized", "baggy", "fitted"]

    category = random.choice(categories)
    if category == "bottom":
        fit = random.choice(bottom_fits)
    elif category in ("top", "outerwear"):
        fit = random.choice(top_fits)
    else:
        fit = None

    return {
        "category": category,
        "color": random.choice(colors),
        "pattern": random.choice(patterns),
        "formality": random.choice(formalities),
        "thickness": random.choice(thicknesses),
        "fit": fit,
        "season": random.choice(seasons_options),
    }


def tag_with_claude(image_bytes: bytes, formality_hint: Optional[str] = None) -> dict:
    import base64

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    prompt = TAGGING_PROMPT
    if formality_hint:
        prompt += f"\n\nThe user describes their idea of 'formal' as: {formality_hint}. Use this to calibrate the formality field."

    response = claude_client.messages.create(
        model="claude-sonnet-5",
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
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Claude returned unparseable tags: {text}")


def tag_with_gemini(image_bytes: bytes, formality_hint: Optional[str] = None) -> dict:
    prompt = TAGGING_PROMPT
    if formality_hint:
        prompt += f"\n\nThe user describes their idea of 'formal' as: {formality_hint}. Use this to calibrate the formality field."

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Gemini returned unparseable tags: {text}")
