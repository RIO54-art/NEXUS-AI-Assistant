import time
import numpy as np
import onnxruntime as ort
import sounddevice as sd

SAMPLE_RATE = 16000
DEVICE = 1
WINDOW = 512

MODEL = r".\.venv\Lib\site-packages\openwakeword\resources\models\silero_vad.onnx"

session = ort.InferenceSession(
    MODEL,
    providers=["CPUExecutionProvider"],
)

h = np.zeros((2, 1, 64), dtype=np.float32)
c = np.zeros((2, 1, 64), dtype=np.float32)


def get_probability(pcm):
    global h, c

    audio = (
        np.frombuffer(pcm, dtype=np.int16)
        .astype(np.float32)
        / 32768.0
    ).reshape(1, WINDOW)

    output, h, c = session.run(
        None,
        {
            "input": audio,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            "h": h,
            "c": c,
        },
    )

    return float(np.asarray(output).squeeze())


print("NEXUS VAD PROBABILITY TEST")
print("Stay SILENT for 3 seconds...")

stream = sd.RawInputStream(
    samplerate=SAMPLE_RATE,
    blocksize=WINDOW,
    device=DEVICE,
    channels=1,
    dtype="int16",
)

stream.start()

silent = []

start = time.time()
while time.time() - start < 3:
    data, _ = stream.read(WINDOW)
    silent.append(get_probability(bytes(data)))

print(
    f"Silence: min={min(silent):.3f}, "
    f"max={max(silent):.3f}, "
    f"avg={sum(silent)/len(silent):.3f}"
)

print("Now SPEAK normally for 3 seconds...")

speech = []

start = time.time()
while time.time() - start < 3:
    data, _ = stream.read(WINDOW)
    speech.append(get_probability(bytes(data)))

stream.stop()
stream.close()

print(
    f"Speech:  min={min(speech):.3f}, "
    f"max={max(speech):.3f}, "
    f"avg={sum(speech)/len(speech):.3f}"
)

print("Test complete.")