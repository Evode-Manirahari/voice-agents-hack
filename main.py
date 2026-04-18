#!/usr/bin/env python3
"""FixGuide AI — Multi-mode on-device voice+vision repair, safety & car agent."""

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
GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]

BANNER = """
╔══════════════════════════════════════════════════════╗
║               🔧  FixGuide AI  🔧                    ║
║     On-device voice + vision repair assistant       ║
╚══════════════════════════════════════════════════════╝
"""

MODES = {
    "1": {
        "name": "🔧  Repair Guide",
        "desc": "Home & workplace repair assistant",
        "system": (
            "You are FixGuide AI, an expert repair assistant for field workers and homeowners. "
            "When shown a repair situation:\n"
            "1. Identify what you see in one sentence.\n"
            "2. Give a SAFETY VERDICT: 'DIY SAFE ✓' or 'CALL A PROFESSIONAL ⚠️'\n"
            "3. If DIY safe, give up to 5 numbered steps, each concise.\n"
            "4. If professional needed, state: type of professional, estimated cost range, urgency.\n"
            "5. Add one safety warning if relevant.\n"
            "Keep total response under 150 words — it will be spoken aloud."
        ),
        "welcome": "Repair Guide ready. Show me what needs fixing.",
        "photo_prompt": "Point camera at the repair area.",
        "voice_prompt": "Describe what needs fixing.",
    },
    "2": {
        "name": "🦺  Safety Inspector",
        "desc": "Construction site hazard detection",
        "system": (
            "You are SafeJob AI, an expert construction and workplace safety inspector. "
            "When shown a work site or task:\n"
            "1. Identify the work environment in one sentence.\n"
            "2. Give a SAFETY VERDICT: 'SAFE TO PROCEED ✓' or 'STOP — HAZARD DETECTED ⚠️'\n"
            "3. List up to 3 specific hazards found (or 'No hazards identified').\n"
            "4. List required PPE for this job (e.g., hard hat, gloves, safety glasses).\n"
            "5. One critical safety rule for this task.\n"
            "Keep total response under 120 words — it will be spoken aloud."
        ),
        "welcome": "Safety Inspector ready. Show me the work site.",
        "photo_prompt": "Point camera at the work area or hazard.",
        "voice_prompt": "Describe the task you are about to perform.",
    },
    "3": {
        "name": "🚗  Car Mechanic",
        "desc": "Vehicle diagnostics and repair guidance",
        "system": (
            "You are MechGuide AI, an expert automotive mechanic assistant. "
            "When shown a vehicle issue:\n"
            "1. Identify the vehicle part or symptom in one sentence.\n"
            "2. Give a REPAIR VERDICT: 'DIY REPAIR ✓' or 'SEE A MECHANIC ⚠️'\n"
            "3. If DIY: give up to 5 numbered repair steps.\n"
            "4. Estimated parts cost range in USD.\n"
            "5. Urgency: 'Drive safely' / 'Fix within X days' / 'Stop driving immediately'.\n"
            "Keep total response under 150 words — it will be spoken aloud."
        ),
        "welcome": "Car Mechanic ready. Show me the problem.",
        "photo_prompt": "Point camera at the car issue — under hood, tire, dashboard, etc.",
        "voice_prompt": "Describe the symptom — what sound, warning light, or what happened.",
    },
}

FOLLOWUP_SYSTEM = (
    "You are a helpful repair and safety assistant. Answer follow-up questions clearly "
    "and briefly. Under 80 words. Will be spoken aloud."
)

WORK_ORDER_SYSTEM = (
    "Generate a concise contractor work order a homeowner can send via text or email. "
    "Include: problem, location hint, urgency. Plain text, under 60 words."
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
        if ret:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, frame)
            return tmp.name
    except Exception:
        pass

    print("\n  📱 Webcam unavailable. AirDrop or save a photo from your phone to Mac,")
    print("     then enter the full path below (or press Enter to skip).\n")
    path = input("  Image path: ").strip().strip("'\"")
    if path and Path(path).exists():
        return path
    return None


# ── AI engines ───────────────────────────────────────────────────────────────

def load_on_device_model():
    try:
        from src.cactus import cactus_init, cactus_log_set_level
        cactus_log_set_level(4)
        return cactus_init(str(WEIGHTS_DIR), None, False)
    except Exception as e:
        print(f"  ⚠️  On-device model unavailable ({e.__class__.__name__}). Using cloud only.")
        return None


def _gemini_generate(contents, system_prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
                contents=contents,
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  ⚠️  {model} quota hit, trying next model...")
                continue
            raise
    return "All Gemini models are rate-limited. Please wait a moment and try again."


def ask_gemini_vision(image_path: str, question: str, system_prompt: str) -> str:
    from google.genai import types
    mime = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return _gemini_generate(
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime),
            types.Part.from_text(text=question or "Analyze this and guide me."),
        ],
        system_prompt=system_prompt,
    )


def ask_gemini_text(question: str, context: str, system_prompt: str) -> str:
    return _gemini_generate(
        contents=[f"Prior context: {context}\n\nQuestion: {question}"],
        system_prompt=system_prompt,
    )


def generate_work_order(analysis: str, question: str) -> str:
    return _gemini_generate(
        contents=[f"Repair assessment: {analysis}\nUser described: {question}"],
        system_prompt=WORK_ORDER_SYSTEM,
    )


# ── session ──────────────────────────────────────────────────────────────────

def run_session(model, mode: dict):
    # Step 1: image
    speak(mode["photo_prompt"] + " Taking photo in 3 seconds.")
    print(f"\n📸 {mode['photo_prompt']} (3 seconds...)")
    time.sleep(3)
    image_path = capture_image()
    if image_path:
        print(f"  ✓ Image ready")
    else:
        print("  ⚠️  No image. Continuing without image.")
        speak("No image. Please describe in detail.")

    # Step 2: voice
    speak(mode["voice_prompt"])
    audio_path = record_voice(seconds=7)
    question = transcribe_audio(audio_path)

    if question:
        print(f'\n  🗣  You said: "{question}"')
    else:
        question = "What do you see? Analyze and guide me."
        print("  (Could not understand audio — using default question)")

    # Step 3: analyze
    print("\n  🤖 Analyzing...")
    if image_path and GEMINI_API_KEY:
        response = ask_gemini_vision(image_path, question, mode["system"])
    elif GEMINI_API_KEY:
        response = ask_gemini_text(question, "", mode["system"])
    else:
        response = "Cloud AI unavailable. Please check your GEMINI_API_KEY."

    print(f"\n{'─'*50}\n{response}\n{'─'*50}\n")
    speak(response)

    # Work order offer if professional needed
    needs_pro = any(w in response.upper() for w in ["CALL A PROFESSIONAL", "SEE A MECHANIC", "STOP — HAZARD"])
    if needs_pro and GEMINI_API_KEY:
        speak("Would you like me to generate a work order you can send to a professional? Say yes or no.")
        audio_path = record_voice(seconds=4)
        answer = transcribe_audio(audio_path).lower()
        if "yes" in answer or "yeah" in answer or "sure" in answer:
            print("  📋 Generating work order...")
            work_order = generate_work_order(response, question)
            print(f"\n  📋 WORK ORDER:\n  {work_order}\n")
            speak("Here is your work order: " + work_order)

    # Step 4: follow-up loop
    session_context = response
    while True:
        speak("Any follow-up questions? Say done to finish.")
        audio_path = record_voice(seconds=6)
        followup = transcribe_audio(audio_path)

        if not followup:
            break

        print(f'\n  🗣  Follow-up: "{followup}"')

        if any(w in followup.lower() for w in ["done", "finish", "stop", "next", "quit", "exit", "no", "thank"]):
            speak("Got it. Session complete.")
            break

        print("  💬 Answering...")
        answer = ask_gemini_text(followup, session_context, FOLLOWUP_SYSTEM)
        print(f"\n  {answer}\n")
        speak(answer)
        session_context += f" {answer}"


# ── main ─────────────────────────────────────────────────────────────────────

def select_mode() -> dict:
    print("\nSelect a mode:\n")
    for key, mode in MODES.items():
        print(f"  [{key}] {mode['name']} — {mode['desc']}")
    print()
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in MODES:
            return MODES[choice]
        print("  Please enter 1, 2, or 3.")


def main():
    print(BANNER)

    if not GEMINI_API_KEY:
        print("⚠️  GEMINI_API_KEY not set — run: export GEMINI_API_KEY='your-key'\n")

    print("Loading on-device model (Gemma 270M)...")
    model = load_on_device_model()
    print("✓ On-device model ready\n" if model else "✓ Running in cloud-only mode (Gemini)\n")

    mode = select_mode()
    print(f"\n  Mode: {mode['name']}\n")
    speak(mode["welcome"])

    while True:
        print("─" * 52)
        print("Press Enter to start a session (m = change mode, q = quit): ", end="", flush=True)
        cmd = input().strip().lower()
        if cmd == "q":
            break
        if cmd == "m":
            mode = select_mode()
            speak(mode["welcome"])
            continue
        run_session(model, mode)

    if model:
        try:
            from src.cactus import cactus_destroy
            cactus_destroy(model)
        except Exception:
            pass
    speak("Stay safe. Goodbye!")
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
