import os
import re
import subprocess
import time
import wave
from collections import deque

import numpy as np
import sounddevice as sd


# =========================================================
# NEXUS VOICE ENGINE
# =========================================================

SAMPLE_RATE = 16000
MICROPHONE_DEVICE = 1

BLOCK_DURATION = 0.02
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)


# =========================================================
# VOICE ACTIVITY DETECTION
# =========================================================

NOISE_FLOOR_INITIAL = 35.0

START_MULTIPLIER = 2.4
STOP_MULTIPLIER = 1.35

MIN_START_LEVEL = 120.0
MIN_STOP_LEVEL = 60.0

START_CONFIRM_BLOCKS = 2

SILENCE_DURATION = 0.45

START_TIMEOUT = 5.0
MAX_RECORD_SECONDS = 20.0
MIN_SPEECH_SECONDS = 0.20

PRE_ROLL_SECONDS = 0.35
PRE_ROLL_BLOCKS = int(
    PRE_ROLL_SECONDS / BLOCK_DURATION
)


# =========================================================
# WHISPER
# =========================================================

WHISPER_EXE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "whisper-bin-x64",
        "Release",
        "whisper-cli.exe",
    )
)

WHISPER_MODEL = r"C:\Users\Anup\Downloads\ggml-base.en.bin"

AUDIO_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "voice_input.wav",
    )
)


# =========================================================
# AUDIO UTILITIES
# =========================================================

def _audio_level(audio):
    """
    Calculate RMS microphone level.
    """

    audio = audio.astype(np.float32)

    if len(audio) == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(audio * audio)
        )
    )


def _smooth_level(previous, current, alpha=0.35):
    """
    Smooth microphone level fluctuations.
    """

    return previous + alpha * (
        current - previous
    )


# =========================================================
# SAVE WAV
# =========================================================

def _write_wav(audio_data):
    """
    Save mono 16 kHz int16 audio.
    """

    with wave.open(AUDIO_FILE, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        wav.writeframes(
            audio_data.astype(
                np.int16
            ).tobytes()
        )


# =========================================================
# RECORD COMMAND FROM EXISTING STREAM
# =========================================================

def record_command_from_stream(
    stream,
    initial_audio=None,
):
    """
    Record RIO's command using an already-open
    microphone stream.

    IMPORTANT:

    This function does NOT open another microphone.

    It continues reading from the same stream that
    detected the wake word.

    initial_audio:
        Optional audio already captured immediately
        after wake-word detection.

        This prevents the wake-word → command handoff
        from losing audio.
    """

    print(
        "\n🎙️ Listening for your command..."
    )

    frames = []

    if initial_audio is not None:
        frames.append(
            np.asarray(
                initial_audio,
                dtype=np.int16
            ).copy()
        )

    pre_roll = deque(
        maxlen=PRE_ROLL_BLOCKS
    )

    speech_started = False

    speech_start_time = None
    last_voice_time = None

    consecutive_voice_blocks = 0

    noise_floor = NOISE_FLOOR_INITIAL

    smoothed_level = noise_floor

    start_time = time.monotonic()

    while True:

        audio, overflowed = stream.read(
            BLOCK_SIZE
        )

        if overflowed:
            print(
                "⚠️ Microphone buffer overflow."
            )

        audio = audio[:, 0].astype(
            np.int16
        )

        level = _audio_level(audio)

        smoothed_level = _smooth_level(
            smoothed_level,
            level,
        )

        now = time.monotonic()

        # =================================================
        # WAITING FOR COMMAND SPEECH
        # =================================================

        if not speech_started:

            pre_roll.append(
                audio.copy()
            )

            # Learn the background noise slowly.
            if level < max(
                noise_floor * 1.6,
                MIN_START_LEVEL,
            ):

                noise_floor = (
                    noise_floor * 0.97
                    + level * 0.03
                )

            start_threshold = max(
                MIN_START_LEVEL,
                noise_floor * START_MULTIPLIER,
            )

            if smoothed_level >= start_threshold:

                consecutive_voice_blocks += 1

            else:

                consecutive_voice_blocks = 0

            if (
                consecutive_voice_blocks
                >= START_CONFIRM_BLOCKS
            ):

                speech_started = True

                speech_start_time = now
                last_voice_time = now

                print(
                    "🎙️ Voice detected."
                )

                # Preserve audio immediately before
                # speech started.
                frames.extend(
                    list(pre_roll)
                )

                frames.append(
                    audio.copy()
                )

            elif (
                now - start_time
                >= START_TIMEOUT
            ):

                print(
                    "🎙️ No command detected."
                )

                return False

        # =================================================
        # COMMAND SPEECH
        # =================================================

        else:

            frames.append(
                audio.copy()
            )

            stop_threshold = max(
                MIN_STOP_LEVEL,
                noise_floor * STOP_MULTIPLIER,
            )

            if (
                smoothed_level
                >= stop_threshold
            ):

                last_voice_time = now

            # Natural end of command.
            if (
                last_voice_time is not None
                and
                now - last_voice_time
                >= SILENCE_DURATION
            ):

                break

            # Safety limit.
            if (
                speech_start_time is not None
                and
                now - speech_start_time
                >= MAX_RECORD_SECONDS
            ):

                print(
                    "⏱️ Maximum command length reached."
                )

                break

    # =========================================================
    # VALIDATE
    # =========================================================

    if not frames:
        return False

    speech_duration = (
        time.monotonic()
        - speech_start_time
        if speech_start_time is not None
        else 0
    )

    if (
        speech_duration
        < MIN_SPEECH_SECONDS
    ):

        print(
            "🎙️ Speech was too short."
        )

        return False

    # =========================================================
    # SAVE AUDIO
    # =========================================================

    audio_data = np.concatenate(
        frames
    )

    _write_wav(
        audio_data
    )

    print(
        "🎙️ Command recording finished."
    )

    return True


# =========================================================
# STANDALONE RECORDING
# =========================================================

def record_audio():
    """
    Standalone microphone recording.

    Used by microphone tests and other parts of NEXUS
    that don't already own an InputStream.
    """

    print(
        "\n🎙️ Waiting for your voice..."
    )

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="int16",
            device=MICROPHONE_DEVICE,
        ) as stream:

            return record_command_from_stream(
                stream
            )

    except Exception as error:

        raise RuntimeError(
            f"Microphone error: {error}"
        ) from error


# =========================================================
# WHISPER TRANSCRIPTION
# =========================================================

def transcribe_audio():
    """
    Local Whisper transcription.
    """

    if not os.path.exists(
        WHISPER_EXE
    ):

        raise FileNotFoundError(
            f"Whisper executable not found:\n"
            f"{WHISPER_EXE}"
        )

    if not os.path.exists(
        WHISPER_MODEL
    ):

        raise FileNotFoundError(
            f"Whisper model not found:\n"
            f"{WHISPER_MODEL}"
        )

    if not os.path.exists(
        AUDIO_FILE
    ):

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{AUDIO_FILE}"
        )

    cpu_count = (
        os.cpu_count()
        or 4
    )

    whisper_threads = max(
        2,
        min(
            cpu_count - 1,
            8,
        ),
    )

    result = subprocess.run(
        [
            WHISPER_EXE,
            "-m",
            WHISPER_MODEL,
            "-f",
            AUDIO_FILE,
            "-t",
            str(whisper_threads),
            "-nt",
            "-np",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr.strip()
            or "Whisper failed."
        )

    return result.stdout.strip()


import threading
from typing import Optional

# =========================================================
# TEXT TO SPEECH (PERSISTENT SAPI WORKER)
# =========================================================

class PersistentTTS:
    """
    Long-lived PowerShell SAPI speech synthesis process.
    Initializes System.Speech once at startup to eliminate
    process creation overhead on every spoken phrase.
    """

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _ensure_running(self):
        if self._proc is not None and self._proc.poll() is None:
            return

        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Rate = 1; "
            "$s.Volume = 100; "
            "while ($true) { "
            "  $line = [Console]::In.ReadLine(); "
            "  if ($null -eq $line -or $line -eq '__NEXUS_EXIT__') { break }; "
            "  if ($line.Trim() -ne '') { $s.Speak($line) }; "
            "  [Console]::Out.WriteLine('DONE'); "
            "  [Console]::Out.Flush(); "
            "}"
        )

        try:
            self._proc = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", ps_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except Exception as err:
            self._proc = None
            print(f"Warning: Persistent TTS startup failed: {err}")

    def speak(self, text: str):
        if not text:
            return

        clean_text = re.sub(r"[^\x00-\x7F]+", "", text)
        clean_text = re.sub(r"[\r\n]+", " ", clean_text).strip()

        if not clean_text:
            return

        with self._lock:
            self._ensure_running()

            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(clean_text + "\n")
                    self._proc.stdin.flush()
                    _ = self._proc.stdout.readline()
                    return
                except Exception as err:
                    print(f"Persistent TTS error: {err}, falling back...")
                    self.stop()

            # Fallback if persistent process unavailable
            _fallback_speak(clean_text)

    def stop(self):
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.poll() is None:
                        self._proc.stdin.write("__NEXUS_EXIT__\n")
                        self._proc.stdin.flush()
                        self._proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None


def _fallback_speak(text: str):
    clean_text = re.sub(r"[^\x00-\x7F]+", "", text).strip()
    if not clean_text:
        return

    command = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 1
$s.Volume = 100
$s.Speak($env:NEXUS_TEXT)
$s.Dispose()
"""
    env = os.environ.copy()
    env["NEXUS_TEXT"] = clean_text
    try:
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            env=env,
            check=True,
        )
    except Exception as err:
        print(f"TTS Fallback error: {err}")


_tts_instance = PersistentTTS()


def speak_text(text: str):
    """
    Speak text using NEXUS persistent Windows SAPI engine.
    """
    _tts_instance.speak(text)


def stop_tts():
    """
    Cleanly shutdown the persistent TTS engine.
    """
    _tts_instance.stop()
