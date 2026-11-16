from __future__ import annotations

import hashlib
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modi_transliterator.crnn import prepare_line_image
from modi_transliterator.line_segmentation import detect_line_regions
from scripts.evaluate_crnn import load_model


CHECKPOINT = PROJECT_ROOT / "outputs" / "crnn_main_seed_2026" / "best.pt"
DEMO_IMAGE = PROJECT_ROOT / "data" / "images" / "mode_00001.png"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
RESULT_VERSION = 2


@st.cache_resource(show_spinner=False)
def load_engine(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, vocabulary, model = load_model(Path(checkpoint_path), device)
    return device, checkpoint, vocabulary, model


def open_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The image is larger than 25 MB. Please upload a smaller copy.")
    try:
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("L")
            image.load()
            return image
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("That file could not be read as an image.") from exc


def ctc_line_confidence(
    line_log_probabilities: torch.Tensor,
    predicted_ids: torch.Tensor,
    blank_id: int,
) -> tuple[float, int]:
    """Return the geometric mean posterior for emitted CTC characters."""
    emitted_log_probabilities: list[torch.Tensor] = []
    previous = blank_id
    for timestep, raw_index in enumerate(predicted_ids.tolist()):
        index = int(raw_index)
        if index != blank_id and index != previous:
            emitted_log_probabilities.append(line_log_probabilities[timestep, index])
        previous = index
    if not emitted_log_probabilities:
        return 0.0, 0
    confidence = torch.stack(emitted_log_probabilities).mean().exp().item()
    return float(confidence), len(emitted_log_probabilities)


def recognize(source: Image.Image) -> dict[str, object]:
    # The training segmenter falls back to a full-image region on empty input.
    # Reject a constant image at the UI boundary before loading the model.
    gray = np.asarray(source.convert("L"))
    if int(gray.max()) - int(gray.min()) < 3:
        raise ValueError("No readable text lines were detected. Upload a manuscript segment with visible writing.")
    device, checkpoint, vocabulary, model = load_engine(str(CHECKPOINT))
    regions = detect_line_regions(source)
    if not regions:
        raise ValueError(
            "No readable text lines were detected. Try a tighter crop with dark Modi writing on a light background."
        )

    height = int(checkpoint["metadata"]["image_height"])
    max_width = int(checkpoint["metadata"]["max_width"])
    prepared = [
        prepare_line_image(source.crop(region.box), height, max_width)
        for region in regions
    ]
    tensors = [item[0] for item in prepared]
    widths = torch.tensor([item[1] for item in prepared], dtype=torch.long)
    widest = max(tensor.shape[-1] for tensor in tensors)
    batch = torch.ones(len(tensors), 1, height, widest)
    for index, tensor in enumerate(tensors):
        batch[index, :, :, : tensor.shape[-1]] = tensor

    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(batch.to(device))
    elapsed = time.perf_counter() - started
    predictions = logits.argmax(dim=-1).transpose(0, 1).cpu()
    output_lengths = model.output_lengths(widths)
    lines: list[str] = []
    confidences: list[float] = []
    emitted_counts: list[int] = []
    for index, row in enumerate(predictions):
        valid_length = min(int(output_lengths[index]), int(logits.shape[0]))
        valid_ids = row[:valid_length]
        lines.append(vocabulary.decode_greedy(valid_ids.tolist()).strip())
        confidence, emitted_count = ctc_line_confidence(
            logits[:valid_length, index].cpu(),
            valid_ids,
            vocabulary.blank_id,
        )
        confidences.append(confidence)
        emitted_counts.append(emitted_count)

    total_emitted = sum(emitted_counts)
    if total_emitted:
        overall_confidence = sum(
            confidence * emitted_count
            for confidence, emitted_count in zip(confidences, emitted_counts)
        ) / total_emitted
    else:
        overall_confidence = 0.0

    overlay = source.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    stroke_width = max(3, overlay.width // 420)
    for index, region in enumerate(regions, start=1):
        draw.rectangle(region.box, outline=(143, 45, 43), width=stroke_width)
        badge_x = region.x1 + stroke_width
        badge_y = region.y1 + stroke_width
        badge_text = str(index)
        badge_width = 16 + (8 * len(badge_text))
        draw.rounded_rectangle(
            (badge_x, badge_y, badge_x + badge_width, badge_y + 22),
            radius=5,
            fill=(143, 45, 43),
        )
        draw.text((badge_x + 7, badge_y + 4), badge_text, fill=(255, 250, 238))

    combined = "\n".join(line for line in lines if line)
    return {
        "overlay": overlay,
        "lines": lines,
        "line_confidences": confidences,
        "combined_text": combined,
        "line_count": len(lines),
        "overall_confidence": overall_confidence,
        "elapsed_seconds": elapsed,
        "device": "GPU" if device.type == "cuda" else "CPU",
    }


def image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def source_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def set_demo_image() -> None:
    if DEMO_IMAGE.exists():
        st.session_state["demo_enabled"] = True


st.set_page_config(
    page_title="Modi Script Reader",
    page_icon="म",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #2f231c;
        --muted: #74655b;
        --paper: #fffaf0;
        --paper-deep: #f5ead5;
        --maroon: #8f2d2b;
        --maroon-dark: #6f211f;
        --gold: #c58a2b;
      }
      .stApp {
        background:
          radial-gradient(circle at 8% 3%, rgba(197,138,43,.11), transparent 28rem),
          linear-gradient(180deg, #fffdf8 0%, #fbf5e9 100%);
        color: var(--ink);
      }
      .block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 4rem; }
      header[data-testid="stHeader"] { background: transparent; }
      .hero {
        border: 1px solid rgba(143,45,43,.16);
        border-radius: 26px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.45rem;
        background: linear-gradient(125deg, rgba(255,250,240,.97), rgba(245,234,213,.92));
        box-shadow: 0 18px 55px rgba(85,55,31,.09);
        position: relative;
        overflow: hidden;
      }
      .hero::after {
        content: "मोडी";
        position: absolute;
        right: 2.2rem;
        top: .25rem;
        color: rgba(143,45,43,.075);
        font-size: 8rem;
        line-height: 1;
        font-weight: 800;
      }
      .eyebrow { color: var(--maroon); font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
      .hero h1 { color: var(--ink); font-family: Georgia, serif; font-size: clamp(2.2rem, 5vw, 4.2rem); line-height: 1.02; margin: .6rem 0 .8rem; max-width: 760px; }
      .hero p { color: var(--muted); font-size: 1.08rem; max-width: 690px; margin: 0; }
      .steps { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.35rem; }
      .steps span { background:rgba(255,255,255,.66); border:1px solid rgba(143,45,43,.14); border-radius:999px; padding:.42rem .72rem; color:#5f4a3d; font-size:.84rem; font-weight:650; }
      [data-testid="stFileUploader"] { background: rgba(255,255,255,.72); border: 1px dashed rgba(143,45,43,.38); border-radius: 18px; padding: .55rem; }
      [data-testid="stImage"] img { border-radius: 16px; border: 1px solid rgba(91,61,40,.14); box-shadow: 0 9px 28px rgba(74,48,30,.08); }
      .stButton > button[kind="primary"] { background: var(--maroon); border-color: var(--maroon); color:white; min-height:3rem; font-weight:750; }
      .stButton > button[kind="primary"]:hover { background:var(--maroon-dark); border-color:var(--maroon-dark); }
      .stButton button, .stDownloadButton button { border-radius: 11px; }
      [data-testid="stMetric"] { background:rgba(255,255,255,.72); border:1px solid rgba(91,61,40,.12); border-radius:14px; padding:.8rem 1rem; }
      .result-title { color:var(--maroon); font-family:Georgia, serif; font-size:2rem; margin:.4rem 0 .1rem; }
      .note { color:var(--muted); font-size:.86rem; }
      .footer-note { text-align:center; color:#89786d; margin-top:2.6rem; font-size:.82rem; }
      @media (max-width: 700px) {
        .hero { padding:1.55rem 1.35rem; }
        .hero::after { display:none; }
      }
    </style>
    <section class="hero">
      <div class="eyebrow">Historical manuscript OCR</div>
      <h1>Modi Script Reader</h1>
      <p>Upload a Modi manuscript segment and convert the detected handwritten lines into editable Devanagari text.</p>
      <div class="steps"><span>1 &nbsp;Upload image</span><span>2 &nbsp;Run recognition</span><span>3 &nbsp;Review & download</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not CHECKPOINT.exists():
    st.error(f"The trained model is missing: {CHECKPOINT}")
    st.stop()

upload_col, demo_col = st.columns([4, 1])
with upload_col:
    uploaded = st.file_uploader(
        "Upload a Modi manuscript image",
        type=("png", "jpg", "jpeg", "tif", "tiff", "webp"),
        help="Best results come from a clear, tightly cropped 3–4 line manuscript segment.",
    )
with demo_col:
    st.write("")
    st.write("")
    if DEMO_IMAGE.exists():
        st.button("Use example", on_click=set_demo_image, width="stretch")

source_data: bytes | None = None
source_name: str | None = None
if uploaded is not None:
    source_data = uploaded.getvalue()
    source_name = uploaded.name
    st.session_state["demo_enabled"] = False
elif st.session_state.get("demo_enabled") and DEMO_IMAGE.exists():
    source_data = DEMO_IMAGE.read_bytes()
    source_name = "Example Modi manuscript"

if source_data is None:
    st.info("Choose a PNG, JPG, TIFF, or WebP image to begin.")
else:
    try:
        source_image = open_image(source_data)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    digest = source_digest(source_data)
    result = st.session_state.get("recognition_result")
    if result and (
        result.get("source_digest") != digest
        or result.get("result_version") != RESULT_VERSION
    ):
        st.session_state.pop("recognition_result", None)
        result = None

    preview_col, action_col = st.columns([1.5, 1], gap="large")
    with preview_col:
        st.image(source_image, caption=source_name, width="stretch")
    with action_col:
        st.subheader("Ready to read")
        st.write("The model will find the text lines and transliterate each line into Devanagari.")
        stat_a, stat_b = st.columns(2)
        stat_a.metric("Width", f"{source_image.width:,} px")
        stat_b.metric("Height", f"{source_image.height:,} px")
        st.caption(f"File size: {len(source_data) / 1_048_576:.2f} MB")
        run_clicked = st.button(
            "Recognize Modi text",
            type="primary",
            width="stretch",
        )
        st.markdown('<p class="note">Processing may take a moment the first time while the model loads.</p>', unsafe_allow_html=True)

    if run_clicked:
        try:
            with st.status("Reading the manuscript…", expanded=True) as status:
                st.write("Detecting handwritten lines")
                st.write("Running Modi-to-Devanagari recognition")
                recognition = recognize(source_image)
                recognition["source_digest"] = digest
                recognition["source_name"] = source_name
                recognition["result_version"] = RESULT_VERSION
                st.session_state["recognition_result"] = recognition
                status.update(label="Recognition complete", state="complete", expanded=False)
            result = recognition
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Recognition could not be completed: {exc}")
            result = None

    if result:
        st.divider()
        st.markdown('<div class="result-title">Recognition result</div>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Lines detected", int(result["line_count"]))
        metric_cols[1].metric(
            "Model confidence",
            f"{float(result['overall_confidence']) * 100:.1f}%",
            help="Average CTC character probability. This estimate is not calibrated against human accuracy.",
        )
        metric_cols[2].metric("Model runtime", f"{float(result['elapsed_seconds']):.2f} s")
        metric_cols[3].metric("Processor", str(result["device"]))

        overlay_col, text_col = st.columns([1.25, 1], gap="large")
        with overlay_col:
            st.image(
                result["overlay"],
                caption="Detected reading order",
                width="stretch",
            )
            st.download_button(
                "Download marked image",
                data=image_bytes(result["overlay"]),
                file_name="modi_detected_lines.png",
                mime="image/png",
                width="stretch",
            )
        with text_col:
            st.subheader("Devanagari output")
            combined_text = str(result["combined_text"])
            st.text_area(
                "Transliterated text",
                value=combined_text,
                height=240,
                help="You can select and copy this text.",
            )
            line_data = [
                {
                    "line": index,
                    "text": line,
                    "confidence": round(float(confidence), 4),
                    "confidence_percent": round(float(confidence) * 100, 1),
                }
                for index, (line, confidence) in enumerate(
                    zip(result["lines"], result["line_confidences"]), start=1
                )
            ]
            download_cols = st.columns(2)
            download_cols[0].download_button(
                "Download text",
                data=combined_text.encode("utf-8"),
                file_name="modi_transliteration.txt",
                mime="text/plain",
                width="stretch",
            )
            download_cols[1].download_button(
                "Download JSON",
                data=json.dumps(
                    {
                        "source": result["source_name"],
                        "line_count": result["line_count"],
                        "overall_confidence": round(float(result["overall_confidence"]), 4),
                        "overall_confidence_percent": round(
                            float(result["overall_confidence"]) * 100, 1
                        ),
                        "lines": line_data,
                        "combined_text": combined_text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
                file_name="modi_transliteration.json",
                mime="application/json",
                width="stretch",
            )
            with st.expander("View line-by-line text"):
                for item in line_data:
                    st.markdown(
                        f"**Line {item['line']} · {item['confidence_percent']:.1f}% confidence**"
                    )
                    st.write(item["text"] or "—")

        st.warning(
            "Research prototype: confidence is the model's uncalibrated character-probability estimate, not a guarantee of correctness. A Modi-script expert should verify important historical readings."
        )

st.markdown(
    '<div class="footer-note">Modi Manuscript Transliteration · CRNN-CTC research prototype</div>',
    unsafe_allow_html=True,
)
