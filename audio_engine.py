"""
core/audio_engine.py  —  NEXUS Audio Pipeline
================================================
Provides:
  MicrophoneStream       – continuous 16 kHz mono PCM from a PortAudio device
  NoiseFloorTracker      – adaptive background energy calibration
  EnergyVAD              – voice activity detector (energy + hangover)
  WakeWordDetector       – OpenWakeWord wrapper around hey_nexus.onnx
  CommandCapture         – post-wake speech capture with pre-roll
  WhisperTranscriber     – subprocess wrapper for whisper-cli.exe

All public constants are exported so wakeword.py and test_pipeline.py
can import them without touching the internals.
"""

from __future__ import annotations

import collections
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

# =========================================================
# DEBUG FLAG  (set DEBUG_AUDIO=1 in environment to enable)
# =========================================================

DEBUG_AUDIO: bool = os.environ.get("DEBUG_AUDIO", "0").strip() == "1"
DEBUG_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "debug_audio"
)

# =========================================================
# CONSTANTS — audio format
# =========================================================

SAMPLE_RATE: int = 16_000          # Hz
CHANNELS: int = 1                  # mono
CHUNK_FRAMES: int = 1_280          # 80 ms per chunk (OWW standard)

# =========================================================
# CONSTANTS — wake-word
# =========================================================

WAKE_THRESHOLD: float = float(os.environ.get("NEXUS_WAKE_THRESHOLD", "0.5"))
WAKE_CONFIRMATIONS: int = int(os.environ.get("NEXUS_WAKE_CONFIRMATIONS", "2"))
WAKE_COOLDOWN: float = float(os.environ.get("NEXUS_WAKE_COOLDOWN", "1.5"))
WAKE_MIN_GAP: float = float(os.environ.get("NEXUS_WAKE_MIN_GAP", "2.0"))
POST_TTS_DEAF_SECONDS: float = float(os.environ.get("NEXUS_POST_TTS_DEAF", "1.5"))

# =========================================================
# CONSTANTS — VAD / command capture
# =========================================================

# How many calibration frames before noise floor is "armed"
NOISE_FLOOR_INIT_FRAMES: int = 15

# Speech onset: energy must exceed this × noise floor
SPEECH_ENERGY_RATIO: float = float(os.environ.get("NEXUS_SPEECH_RATIO", "4.0"))

# Hangover: how many silent frames before we declare end-of-speech
HANGOVER_FRAMES: int = int(os.environ.get("NEXUS_HANGOVER_FRAMES", "12"))

# Pre-roll: frames of audio kept before speech onset (catches word beginnings)
PREROLL_FRAMES: int = int(os.environ.get("NEXUS_PREROLL_FRAMES", "6"))

# Post-roll: extra frames recorded after speech ends (catches word endings)
POSTROLL_FRAMES: int = int(os.environ.get("NEXUS_POSTROLL_FRAMES", "4"))

# Minimum speech frames required to keep a capture (avoids tiny bursts)
MIN_SPEECH_FRAMES: int = int(os.environ.get("NEXUS_MIN_SPEECH_FRAMES", "4"))

# How long to wait for the user to start speaking after wake detection
COMMAND_START_TIMEOUT: float = float(os.environ.get("NEXUS_CMD_START_TIMEOUT", "3.0"))

# Maximum total command duration in seconds
COMMAND_MAX_SECONDS: float = float(os.environ.get("NEXUS_CMD_MAX_SECONDS", "15.0"))

# Whisper CLI flags for short English command recognition
WHISPER_FLAGS: list[str] = [
    "--language", "en",
    "--no-timestamps",
    "--beam-size", "5",
    "--best-of", "5",
    "--word-thold", "0.01",
]


# =========================================================
# DEBUG HELPERS
# =========================================================

def _debug_log(msg: str) -> None:
    if DEBUG_AUDIO:
        ts = time.strftime("%H:%M:%S")
        print(f"[DEBUG {ts}] {msg}", flush=True)


def _audio_diagnostics(audio: np.ndarray, label: str) -> None:
    """Print and (if DEBUG_AUDIO) save diagnostics for a captured audio array."""
    if len(audio) == 0:
        print(f"[DIAG] {label}: EMPTY audio")
        return

    duration = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = int(np.max(np.abs(audio)))
    # Speech ratio: fraction of samples above a mild energy threshold
    threshold = max(200, rms * 0.5)
    speech_ratio = float(np.mean(np.abs(audio) > threshold))

    if DEBUG_AUDIO:
        print(f"[DIAG] {label}:", flush=True)
        print(f"       Samples  : {len(audio)}")
        print(f"       Duration : {duration:.2f}s")
        print(f"       RMS      : {rms:.1f}")
        print(f"       Peak     : {peak}")
        print(f"       Speech%  : {speech_ratio:.1%}")

        if rms < 50:
            print("       WARNING  : audio suspiciously quiet (possible silence or mic issue)")
        if peak >= 32700:
            print("       WARNING  : audio is clipping")
        if speech_ratio < 0.05:
            print("       WARNING  : very low speech ratio — may be silence")

        # Save debug WAV
        os.makedirs(DEBUG_DIR, exist_ok=True)
        ts = time.strftime("%H%M%S")
        path = os.path.join(DEBUG_DIR, f"{label}_{ts}.wav")
        _write_wav(path, audio)
        print(f"       Saved    : {path}")


# =========================================================
# WAV WRITER
# =========================================================

def _write_wav(path: str, audio: np.ndarray) -> None:
    """Write int16 numpy array as 16 kHz mono PCM WAV."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.astype(np.int16).tobytes())


# =========================================================
# MICROPHONE STREAM
# =========================================================

class MicrophoneStream:
    """
    Continuous non-blocking microphone capture via sounddevice.

    Reads arrive as 16 kHz mono int16 numpy arrays of length CHUNK_FRAMES.
    Uses a queue with a bounded size to avoid unbounded memory growth when
    the consumer falls behind.
    """

    _QUEUE_MAXSIZE = 64  # ~5 s of audio before we start dropping

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=self._QUEUE_MAXSIZE
        )
        self._stream = None
        self._running = False

    # ----------------------------------------------------------

    def _callback(self, indata, frames, time_info, status):  # noqa: N803
        if status and DEBUG_AUDIO:
            _debug_log(f"MicrophoneStream status: {status}")
        chunk = indata[:, 0].copy()  # [CHUNK_FRAMES] int16
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Drop oldest frame rather than blocking
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    # ----------------------------------------------------------

    def start(self) -> None:
        import sounddevice as sd  # late import – not needed for unit tests

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_FRAMES,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True
        _debug_log(f"MicrophoneStream started (device={self._device})")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        _debug_log("MicrophoneStream stopped")

    def read_chunk(self, timeout: float = 0.15) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard all buffered audio frames (call after TTS to avoid re-triggering)."""
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        _debug_log(f"MicrophoneStream drained {drained} stale frames")


# =========================================================
# NOISE FLOOR TRACKER
# =========================================================

class NoiseFloorTracker:
    """
    Adaptive background energy tracker.

    Uses a running minimum of per-chunk RMS values, smoothed with a
    leaky integrator.  After NOISE_FLOOR_INIT_FRAMES chunks the tracker
    is 'armed' and the floor value is usable.
    """

    _ALPHA_RISE = 0.05   # slow adaptation upward (background noise increase)
    _ALPHA_FALL = 0.20   # faster adaptation downward (quieter environment)

    def __init__(self) -> None:
        self.floor: float = 0.0
        self._count: int = 0
        self._init_values: list[float] = []

    @property
    def armed(self) -> bool:
        return self._count >= NOISE_FLOOR_INIT_FRAMES

    # ----------------------------------------------------------

    def update(self, chunk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

        if not self.armed:
            self._init_values.append(rms)
            self._count += 1
            if self.armed:
                # Initialise floor as the lower percentile of calibration samples
                # to avoid a burst of loud audio biasing it upward.
                self.floor = float(np.percentile(self._init_values, 20))
                _debug_log(f"NoiseFloorTracker armed: floor={self.floor:.1f}")
            return

        # Adaptive leaky integrator
        if rms < self.floor:
            alpha = self._ALPHA_FALL
        else:
            alpha = self._ALPHA_RISE

        self.floor = (1.0 - alpha) * self.floor + alpha * rms


# =========================================================
# ENERGY VAD
# =========================================================

class EnergyVAD:
    """
    Simple energy-based Voice Activity Detector.

    States:
      IDLE    – waiting for speech onset
      SPEECH  – recording speech
      HANGOVER– grace period after speech drops below threshold

    Events returned by feed():
      'start' – speech onset detected
      'end'   – speech ended (after hangover)
      None    – no state change

    Call flush() after 'end' to retrieve the captured audio.
    """

    _IDLE = 0
    _SPEECH = 1
    _HANGOVER = 2

    def __init__(self, noise_tracker: NoiseFloorTracker) -> None:
        self._noise = noise_tracker
        self._state = self._IDLE
        self._hangover_count = 0
        self._speech_frame_count = 0

        # Ring buffer for pre-roll
        self._preroll: collections.deque[np.ndarray] = collections.deque(
            maxlen=PREROLL_FRAMES
        )

        # Captured audio frames (including pre-roll)
        self._capture: list[np.ndarray] = []

        # Post-roll state
        self._postroll_count = 0

    # ----------------------------------------------------------

    def feed(self, chunk: np.ndarray) -> str | None:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        threshold = max(50.0, self._noise.floor * SPEECH_ENERGY_RATIO)
        is_speech = rms > threshold

        if self._state == self._IDLE:
            self._preroll.append(chunk)

            if is_speech:
                # Speech onset — include pre-roll in capture
                self._capture = list(self._preroll)
                self._preroll.clear()
                self._speech_frame_count = 1
                self._state = self._SPEECH
                _debug_log(
                    f"VAD: speech onset (rms={rms:.0f}, thr={threshold:.0f})"
                )
                return "start"

        elif self._state == self._SPEECH:
            self._capture.append(chunk)
            self._speech_frame_count += 1

            if not is_speech:
                self._hangover_count = 1
                self._state = self._HANGOVER
                _debug_log("VAD: entering hangover")

        elif self._state == self._HANGOVER:
            self._capture.append(chunk)

            if is_speech:
                # Speech resumed — go back to SPEECH
                self._hangover_count = 0
                self._state = self._SPEECH
                _debug_log("VAD: speech resumed")
            else:
                self._hangover_count += 1

                if self._hangover_count >= HANGOVER_FRAMES:
                    _debug_log(
                        f"VAD: speech ended after {self._speech_frame_count} "
                        f"speech frames"
                    )
                    self._state = self._IDLE
                    self._hangover_count = 0
                    return "end"

        return None

    # ----------------------------------------------------------

    def flush(self) -> np.ndarray:
        """Return and clear the captured audio (call after 'end' event)."""
        if not self._capture:
            return np.array([], dtype=np.int16)

        audio = np.concatenate(self._capture).astype(np.int16)
        self._capture = []
        self._speech_frame_count = 0
        self._preroll.clear()
        return audio

    def reset(self) -> None:
        """Reset VAD state machine (call before entering command capture)."""
        self._state = self._IDLE
        self._hangover_count = 0
        self._speech_frame_count = 0
        self._capture = []
        self._preroll.clear()
        _debug_log("VAD: reset")


# =========================================================
# WAKE WORD DETECTOR
# =========================================================

class WakeWordDetector:
    """
    OpenWakeWord-based wake-word detector for hey_nexus.onnx.

    Wraps the OWW Model class which handles:
    - mel-spectrogram feature extraction
    - sliding-window inference
    - score accumulation

    feed(chunk) → True when wake word is detected with high confidence.

    The post_wake_buffer holds the audio chunks that were processed
    during/after the wake event so CommandCapture can use them as
    pre-roll (to avoid the same audio being missed or re-processed).
    """

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Wake-word model not found: {model_path}")

        self._model_path = model_path
        self._model = None            # lazily initialised
        self._last_wake_time: float = -WAKE_MIN_GAP
        self._deaf_until: float = 0.0
        self._confirm_count: int = 0

        # Accumulate audio that was fed *after* wake detection (for pre-roll hand-off)
        self.post_wake_buffer: list[np.ndarray] = []

        # Score history for debug
        self._last_score: float = 0.0

        self._init_model()

    # ----------------------------------------------------------

    def _init_model(self) -> None:
        """Load OpenWakeWord model. Raises ImportError if OWW is not installed."""
        try:
            from openwakeword.model import Model as OWWModel  # type: ignore

            # OWW Model with our custom ONNX
            self._model = OWWModel(
                wakeword_models=[self._model_path],
                inference_framework="onnx",
            )
            _debug_log(f"WakeWordDetector: OWW model loaded ({self._model_path})")

        except ImportError:
            # Fall back to direct ONNX inference with our own mel computation
            _debug_log("WakeWordDetector: OWW not installed, using direct ONNX inference")
            self._model = None
            self._init_direct_onnx()

    def _init_direct_onnx(self) -> None:
        """Fallback: direct ONNX inference with manual mel-spectrogram computation."""
        import onnxruntime as ort  # type: ignore

        self._sess = ort.InferenceSession(self._model_path)
        _debug_log("WakeWordDetector: direct ONNX session loaded")

        # Rolling mel buffer: keeps last 16 frames of 96-bin mel features
        self._mel_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=16
        )
        self._mel_fb = self._build_mel_filterbank()

    # ----------------------------------------------------------

    @staticmethod
    def _build_mel_filterbank(
        sr: int = SAMPLE_RATE,
        n_fft: int = 512,
        n_mels: int = 96,
        fmin: float = 0.0,
        fmax: float = 8000.0,
    ) -> np.ndarray:
        """Compute a triangular mel filterbank matching OWW's feature extractor."""

        def hz_to_mel(hz: float) -> float:
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])
        bin_pts = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            lo, mid, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
            for k in range(lo, mid):
                if mid > lo:
                    fb[m - 1, k] = (k - lo) / (mid - lo)
            for k in range(mid, hi):
                if hi > mid:
                    fb[m - 1, k] = (hi - k) / (hi - mid)
        return fb

    # ----------------------------------------------------------

    def _direct_onnx_score(self, chunk: np.ndarray) -> float:
        """Compute wake score via manual mel extraction + direct ONNX inference."""
        # chunk is int16, 1280 samples
        audio = chunk.astype(np.float32) / 32768.0
        n_fft = 512
        hop = 160
        window = np.hanning(n_fft)

        # Extract mel frames from this chunk
        for i in range(0, len(audio) - n_fft, hop):
            frame = audio[i : i + n_fft] * window
            spec = np.abs(np.fft.rfft(frame, n_fft)) ** 2
            mel = np.dot(self._mel_fb, spec)
            mel = np.log(mel + 1e-8).astype(np.float32)
            self._mel_buffer.append(mel)

        if len(self._mel_buffer) < 16:
            return 0.0

        feat = np.array(self._mel_buffer)[np.newaxis]  # [1, 16, 96]
        out = self._sess.run(None, {"x": feat.astype(np.float32)})
        return float(out[0][0][0])

    # ----------------------------------------------------------

    def feed(self, chunk: np.ndarray) -> bool:
        """
        Feed one audio chunk (CHUNK_FRAMES int16 samples).

        Returns True when wake word is detected with required confirmations.
        """
        now = time.monotonic()

        if now < self._deaf_until:
            return False

        # --- Get score ---
        if self._model is not None:
            # OWW Model.predict() returns dict: {model_name: [score_per_chunk]}
            pred = self._model.predict(chunk)
            # Get the score for our model (key is the filename stem or path)
            scores = list(pred.values())
            score: float = float(scores[0]) if scores else 0.0
        else:
            score = self._direct_onnx_score(chunk)

        self._last_score = score

        if DEBUG_AUDIO and score > 0.2:
            _debug_log(f"WakeWordDetector: score={score:.4f}")

        # --- Confirmation logic ---
        if score >= WAKE_THRESHOLD:
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        if self._confirm_count >= WAKE_CONFIRMATIONS:
            # Check minimum gap since last wake
            if now - self._last_wake_time < WAKE_MIN_GAP:
                _debug_log(
                    f"WakeWordDetector: detection suppressed (too soon: "
                    f"{now - self._last_wake_time:.2f}s < {WAKE_MIN_GAP}s)"
                )
                self._confirm_count = 0
                return False

            self._confirm_count = 0
            self._last_wake_time = now
            self.post_wake_buffer = []   # reset hand-off buffer
            _debug_log(
                f"WakeWordDetector: DETECTED (score={score:.4f}, "
                f"confirmations={WAKE_CONFIRMATIONS})"
            )
            return True

        return False

    # ----------------------------------------------------------

    def reset(self) -> None:
        """Reset confirmation counter and OWW internal state."""
        self._confirm_count = 0
        if self._model is not None:
            try:
                # OWW maintains frame history — reset it
                self._model.reset()
            except AttributeError:
                # Older OWW versions may not have reset()
                pass
        else:
            self._mel_buffer.clear()
        _debug_log("WakeWordDetector: reset")

    def set_deaf_until(self, until: float) -> None:
        """Suppress detection until the given monotonic time (post-TTS cooldown)."""
        self._deaf_until = until
        _debug_log(f"WakeWordDetector: deaf for {until - time.monotonic():.1f}s")

    @property
    def last_score(self) -> float:
        return self._last_score


# =========================================================
# COMMAND CAPTURE
# =========================================================

class CommandCapture:
    """
    Captures a spoken command after wake detection.

    Uses EnergyVAD to find speech boundaries, with:
    - A pre-roll already baked into the VAD
    - A start timeout (if user doesn't speak, abandon)
    - A maximum command duration cap
    - The post_wake_chunks from WakeWordDetector used as initial audio
      to fill the VAD pre-roll (so the start of the command isn't cut off)
    """

    def __init__(
        self,
        mic: MicrophoneStream,
        noise_tracker: NoiseFloorTracker,
        timeout_seconds: float = COMMAND_START_TIMEOUT,
    ) -> None:
        self._mic = mic
        self._noise = noise_tracker
        self._timeout = timeout_seconds
        self._vad = EnergyVAD(noise_tracker)

    # ----------------------------------------------------------

    def capture(
        self, post_wake_chunks: list[np.ndarray] | None = None
    ) -> np.ndarray | None:
        """
        Wait for and capture one spoken command.

        Returns int16 numpy array of speech audio, or None if nothing captured.
        """
        self._vad.reset()

        in_speech = False
        speech_found = False
        start_deadline = time.monotonic() + self._timeout
        max_deadline = time.monotonic() + COMMAND_MAX_SECONDS

        # Feed post-wake audio to pre-fill the VAD pre-roll buffer.
        # These chunks were captured DURING/AFTER the wake phrase.
        # They may contain the start of the user's command.
        if post_wake_chunks:
            for chunk in post_wake_chunks:
                event = self._vad.feed(chunk)
                if event == "start":
                    in_speech = True
                    speech_found = True
                elif event == "end" and in_speech:
                    audio = self._vad.flush()
                    if self._valid_audio(audio):
                        _audio_diagnostics(audio, "command_post_wake")
                        return audio
                    in_speech = False

        # Main capture loop — read from the live mic stream
        while time.monotonic() < max_deadline:
            now = time.monotonic()

            if not in_speech and now > start_deadline:
                _debug_log(
                    "CommandCapture: start timeout — no speech detected"
                )
                return None

            chunk = self._mic.read_chunk(timeout=0.15)
            if chunk is None:
                continue

            self._noise.update(chunk)
            event = self._vad.feed(chunk)

            if event == "start":
                in_speech = True
                speech_found = True
                # Reset the start deadline — speech has begun
                _debug_log("CommandCapture: speech onset")

            elif event == "end" and in_speech:
                audio = self._vad.flush()
                _debug_log(
                    f"CommandCapture: speech ended, audio={len(audio)} samples"
                )
                if self._valid_audio(audio):
                    _audio_diagnostics(audio, "command")
                    return audio
                else:
                    _debug_log("CommandCapture: audio too short/quiet, waiting")
                    in_speech = False
                    speech_found = False
                    start_deadline = time.monotonic() + self._timeout

        _debug_log("CommandCapture: max duration reached")
        if in_speech:
            audio = self._vad.flush()
            if self._valid_audio(audio):
                _audio_diagnostics(audio, "command_maxdur")
                return audio

        return None

    # ----------------------------------------------------------

    @staticmethod
    def _valid_audio(audio: np.ndarray) -> bool:
        """Return True if the audio array is worth sending to Whisper."""
        if audio is None or len(audio) == 0:
            return False
        # Minimum 0.4 seconds
        if len(audio) < SAMPLE_RATE * 0.4:
            return False
        # Must have some energy
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < 30.0:
            _debug_log(f"CommandCapture: audio rejected (RMS={rms:.1f} too low)")
            return False
        return True


# =========================================================
# WHISPER TRANSCRIBER
# =========================================================

class WhisperTranscriber:
    """
    Transcribes audio via whisper-cli.exe subprocess.

    Validates audio quality before invocation and reports diagnostics.
    """

    def __init__(self, exe_path: str, model_path: str) -> None:
        if not os.path.isfile(exe_path):
            raise FileNotFoundError(
                f"whisper-cli.exe not found: {exe_path}"
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Whisper model not found: {model_path}"
            )
        self._exe = exe_path
        self._model = model_path

    # ----------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe int16 audio. Returns the stripped transcript string
        (empty string if nothing recognised).
        """
        if audio is None or len(audio) == 0:
            _debug_log("WhisperTranscriber: empty audio, skipping")
            return ""

        # --- Audio quality gate ---
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        peak = int(np.max(np.abs(audio)))
        threshold = max(50.0, rms * 0.3)
        speech_ratio = float(np.mean(np.abs(audio) > threshold))

        if DEBUG_AUDIO:
            print(f"[WHISPER] duration={duration:.2f}s  rms={rms:.0f}  "
                  f"peak={peak}  speech%={speech_ratio:.1%}", flush=True)

        if rms < 30.0:
            print(
                f"[WHISPER] WARNING: Audio RMS={rms:.1f} is suspiciously quiet. "
                "Check microphone gain / device selection.",
                flush=True,
            )

        if peak >= 32700:
            print("[WHISPER] WARNING: Audio is clipping (peak near 32767).", flush=True)

        if speech_ratio < 0.05:
            print(
                f"[WHISPER] WARNING: Very low speech ratio ({speech_ratio:.1%}). "
                "Audio may be silence or background noise.",
                flush=True,
            )
            if rms < 50.0:
                return ""

        # --- Write temp WAV ---
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = tf.name
        _write_wav(wav_path, audio)

        # --- Invoke whisper-cli ---
        cmd = [self._exe, "-m", self._model] + WHISPER_FLAGS + [wav_path]
        _debug_log(f"WhisperTranscriber: {' '.join(cmd)}")

        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            elapsed = time.monotonic() - t0
            _debug_log(f"WhisperTranscriber: done in {elapsed:.2f}s")

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode != 0:
                _debug_log(f"WhisperTranscriber: non-zero exit {result.returncode}")
                if DEBUG_AUDIO and stderr:
                    print(f"[WHISPER stderr] {stderr[:500]}", flush=True)

            transcript = self._clean_transcript(stdout)
            _debug_log(f"WhisperTranscriber: transcript={repr(transcript)}")
            return transcript

        except subprocess.TimeoutExpired:
            print("[WHISPER] ERROR: whisper-cli timed out after 60s", flush=True)
            return ""

        except Exception as exc:
            print(f"[WHISPER] ERROR: {exc}", flush=True)
            return ""

        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    # ----------------------------------------------------------

    @staticmethod
    def _clean_transcript(raw: str) -> str:
        """
        Strip whisper-cli formatting noise from the output.

        whisper-cli emits lines like:
          [00:00:00.000 --> 00:00:01.000]  Hello world.
        or bare text when --no-timestamps is used.
        """
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove timestamp brackets  [HH:MM:SS.mmm --> HH:MM:SS.mmm]
            if line.startswith("[") and "-->" in line and "]" in line:
                idx = line.index("]")
                line = line[idx + 1 :].strip()
            # Skip whisper system output lines
            if any(
                line.startswith(p)
                for p in (
                    "whisper_",
                    "ggml_",
                    "main:",
                    "system_info:",
                    "sampling:",
                    "log_mel_spectrogram",
                )
            ):
                continue
            if line:
                lines.append(line)

        transcript = " ".join(lines).strip()

        # Remove common hallucination patterns whisper produces on silence
        hallucinations = {
            "[BLANK_AUDIO]",
            "(BLANK_AUDIO)",
            "[silence]",
            "(silence)",
            "[ Silence ]",
            "[Music]",
            "(Music)",
            "Thank you.",
            "Thank you for watching.",
            "Thank you for watching!",
            "Thank you. Bye.",
        }
        if transcript in hallucinations:
            return ""

        return transcript
