#!/usr/bin/env python3
"""FixGuide AI — On-device voice + vision repair agent."""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "cactus" / "python"))

WEIGHTS_DIR = REPO_ROOT / "cactus" / "weights" / "functiongemma-270m-it"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

BANNER = """
╔══════════════════════════════════════════════╗
║           🔧  FixGuide AI  🔧                ║
║  On-device voice + vision repair assistant  ║
╚══════════════════════════════════════════════╝
"""

SYSTEM_PROMPT = (
    "You are FixGuide AI, an expert repair assistant for field workers and homeowners. "
    "When shown a repair situation:\n"
    "1. Identify what you see in one sentence.\n"
    "2. Give a SAFETY VERDICT: 'DIY SAFE ✓' or 'CALL A PROFESSIONAL ⚠️'\n"
    "3. If DIY safe, give up to 5 numbered steps, each concise.\n"
    "4. Add one safety warning if relevant.\n"
    "Keep total response under 120 words — it will be spoken aloud."
)

FOLLOWUP_PROMPT = (
    "You are FixGuide AI. Answer follow-up repair questions clearly and briefly. "
    "Under 80 words. This will be spoken aloud."
)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def speak(text: str):
    clean = text.replace('"', "'").replace("\n", " ")
    subprocess.run(["say", "-r", "175", clean], check=False)


def record_voice(seconds: int = 7) -> str:
    import sounddevice as sd
    import soundfile as sf
    sample_rate = 16000
    print(f"  🎤 Listening for {seconds}s...", flush=True)
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def transcribe_audio(audio_path: str) -> str:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except Exception:
        return ""


def capture_image() -> str | None:
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        time.sleep(0.8)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, frame)
        return tmp.name
    except Exception:
        return None


# ── AI engines ───────────────────────────────────────────────────────────────

def load_on_device_model():
    from src.cactus import cactus_init, cactus_log_set_level
    cactus_log_set_level(4)
    return cactus_init(str(WEIGHTS_DIR), None, False)


def ask_on_device(model, question: str) -> str:
    from src.cactus import cactus_complete
    messages = json.dumps([{"role": "user", "content": question}])
    try:
        raw = cactus_complete(model, messages, None, None, None)
        data = json.loads(raw)
        return data.get("response", "").strip()
    except Exception:
        return ""


def ask_gemini_vision(image_path: str, question: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=question or "What needs fixing here? Guide me."),
        ],
    )
    return response.text.strip()


def ask_gemini_text(question: str, context: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction=FOLLOWUP_PROMPT),
        contents=[f"Prior context: {context}\n\nFollow-up question: {question}"],
    )
    return response.text.strip()


# ── main loop ────────────────────────────────────────────────────────────────

def run_repair_session(model):
    # Step 1: capture image
    speak("Point your camera at the repair area. Taking photo in 3 seconds.")
    print("\n📸 Taking photo in 3 seconds — point camera at the problem area...")
    time.sleep(3)
    image_path = capture_image()
    if image_path:
        print(f"  ✓ Image captured")
    else:
        print("  ⚠️  No camera found. Continuing without image.")
        speak("No camera found. Please describe the problem.")

    # Step 2: voice input
    speak("Now describe what needs fixing.")
    audio_path = record_voice(seconds=7)
    question = transcribe_audio(audio_path)

    if question:
        print(f'\n  🗣  You said: "{question}"')
    else:
        question = "What do you see here that needs repair? How do I fix it safely?"
        print("  (Could not understand audio — using default question)")

    # Step 3: analyze
    print("\n  🤖 Analyzing...")
    if image_path and GEMINI_API_KEY:
        response = ask_gemini_vision(image_path, question)
        engine = "Gemini Vision + on-device routing"
    elif GEMINI_API_KEY:
        response = ask_gemini_text(question, "")
        engine = "Gemini (no image)"
    else:
        response = ask_on_device(model, question)
        engine = "On-device Gemma"

    print(f"\n  [{engine}]\n")
    print(f"  {response}\n")
    speak(response)

    # Step 4: follow-up loop
    session_context = response
    while True:
        speak("Any follow-up questions? Say done to start a new repair.")
        audio_path = record_voice(seconds=6)
        followup = transcribe_audio(audio_path)

        if not followup:
            print("  (No follow-up detected)")
            break

        print(f'\n  🗣  Follow-up: "{followup}"')

        if any(w in followup.lower() for w in ["done", "finish", "stop", "next", "quit", "exit"]):
            break

        print("  💬 Answering...")
        if GEMINI_API_KEY:
            answer = ask_gemini_text(followup, session_context)
        else:
            answer = ask_on_device(model, f"Context: {session_context}\n\nQuestion: {followup}")

        print(f"\n  {answer}\n")
        speak(answer)
        session_context += f" {answer}"


def main():
    print(BANNER)

    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not set — will use on-device model only")
        print("   Run: export GEMINI_API_KEY='your-key'\n")

    print("Loading on-device model (Gemma 270M)...")
    model = load_on_device_model()
    print("✓ Model ready\n")

    speak("Fix Guide ready. Let's get your repair done safely.")

    while True:
        print("─" * 48)
        print("Press Enter to start a repair session (q to quit): ", end="", flush=True)
        cmd = input().strip().lower()
        if cmd == "q":
            break
        run_repair_session(model)

    from src.cactus import cactus_destroy
    cactus_destroy(model)
    speak("Stay safe. Goodbye!")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
