from flask import Flask, Response, stream_with_context, request
from typing import Iterable
from voxcpm import VoxCPM
from voxcpm.utils.text_normalize import TextNormalizer
from realtime_codec_agent.audio_tokenizer import AudioTokenizer
import numpy as np
import torch
import re
import argparse
import base64
import tempfile

app = Flask(__name__)
voxcpm_model: VoxCPM = None
audio_tokenizer: AudioTokenizer = None
text_normalizer: TextNormalizer = None

session_prompt_caches = {}
pause_regex = re.compile(r"\(\d*?\.\d*?\)")

def _sanitize_text_for_tts(text):
    text = re.sub(pause_regex, "...", text)
    text = re.sub(r"(?:\s|\A)i?[hx]+[.,?!]*(?=(?:\s|\Z))", "", text, flags=re.IGNORECASE)
    text = re.sub(r"0 ?(?=\[)", "", text)
    text = re.sub("0[.]", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"&=.*?(?=(?:\s|\Z))", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

@stream_with_context
def generate_chunks(sid: str, text: str, chunk_size_secs: float) -> Iterable[str]:
    sid_prompt_cache = session_prompt_caches.get(sid)
    if sid_prompt_cache is None:
        sid_prompt_cache = {
            "fixed_prompt_cache": None,
            "dynamic_prompt_cache": None,
        }
        session_prompt_caches[sid] = sid_prompt_cache
    fixed_prompt_cache, dynamic_prompt_cache = sid_prompt_cache["fixed_prompt_cache"], sid_prompt_cache["dynamic_prompt_cache"]

    text = _sanitize_text_for_tts(text)
    if not text:
        return
    text = text_normalizer.normalize(text)
    stream = voxcpm_model.tts_model.generate_with_prompt_cache_streaming(
        target_text=text,
        prompt_cache=fixed_prompt_cache,#dynamic_prompt_cache if dynamic_prompt_cache is not None else fixed_prompt_cache,
        inference_timesteps=5,
    )
    chunk_size_samples = int(chunk_size_secs * voxcpm_model.tts_model.sample_rate)
    buffer = np.zeros((0,), dtype=np.float32)
    for wav, target_text_token, generated_audio_feat in stream:
        wav = wav.squeeze(0).cpu().numpy()
        buffer = np.concatenate((buffer, wav), axis=-1)
        if buffer.shape[-1] >= chunk_size_samples:
            chunk, buffer = np.split(buffer, [chunk_size_samples], axis=-1)
            chunk_str = audio_tokenizer.tokenize_audio(chunk)
            yield chunk_str + "\n"
    # update the prompt cache to be consistent with the generated audio
    generated_audio_feat = torch.cat(generated_audio_feat, dim=1).squeeze(0).cpu()
    merged_prompt_cache = voxcpm_model.tts_model.merge_prompt_cache(
        original_cache=fixed_prompt_cache,
        new_text_token=target_text_token,
        new_audio_feat=generated_audio_feat,
    )
    if fixed_prompt_cache is None:
        sid_prompt_cache["fixed_prompt_cache"] = merged_prompt_cache
    else:
        sid_prompt_cache["dynamic_prompt_cache"] = merged_prompt_cache

@app.route("/stream", methods=["POST"])
def stream_chunks() -> Response:
    data = request.get_json(force=True)
    sid = data.get("session_id")
    if not sid:
        return Response(
            "No session_id provided. Generate a unique identifier and provide it in the session_id field.", 
            status=400,
        )
    text = data.get("text", "")
    chunk_size_secs = float(data.get("chunk_size_secs", 0.1))
    return Response(generate_chunks(sid, text, chunk_size_secs), mimetype="text/plain")

@app.route("/set_voice_enrollment", methods=["POST"])
def set_voice_enrollment():
    data = request.get_json(force=True)
    sid = data.get("session_id")
    if not sid:
        return Response(
            "No session_id provided. Generate a unique identifier and provide it in the session_id field.", 
            status=400,
        )
    b64_wav = data.get("wav_base64")
    if not b64_wav:
        session_prompt_caches.pop(sid, None)
        return Response(status=200)
    prompt_text = data.get("prompt_text", "")
    if prompt_text:
        prompt_text = prompt_text.strip()
    if not prompt_text:
        return Response("No prompt_text provided", status=400)
    try:
        wav_bytes = base64.b64decode(b64_wav)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(wav_bytes)
            tmp_path = tmp_file.name
        fixed_prompt_cache = voxcpm_model.tts_model.build_prompt_cache(
            prompt_wav_path=tmp_path, 
            prompt_text=prompt_text,
        )
        session_prompt_caches[sid] = {
            "fixed_prompt_cache": fixed_prompt_cache,
            "dynamic_prompt_cache": None,
        }
        return Response(status=200)
    except Exception as e:
        return Response(f"Error processing file: {str(e)}", status=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Realtime Codec Agent with FastRTC.")
    parser.add_argument(
        "--tts_model", 
        default="openbmb/VoxCPM-0.5B", 
        help="Path to the VoxCPM model.",
    )
    parser.add_argument(
        "--codec_model",
        default="MagiCodec-50Hz-Base",
        help="Path to the MagiCodec model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable optimizations that could interfere with debugging."
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    voxcpm_model = VoxCPM.from_pretrained(
        args.tts_model, 
        load_denoiser=False, 
        optimize=not args.debug,
    )
    audio_tokenizer = AudioTokenizer(codec_model=args.codec_model)
    text_normalizer = TextNormalizer()

    # Development server; use a production WSGI server (gunicorn, etc.) for prod.
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
