import sounddevice as sd
from openwakeword.model import Model

model_path = r".\.venv\Lib\site-packages\openwakeword\resources\models\hey_jarvis_v0.1.onnx"
model = Model(wakeword_models=[model_path], inference_framework="onnx")

print("LISTENING - say Hey Jarvis for 10 seconds")

audio = sd.rec(
    160000,
    samplerate=16000,
    channels=1,
    dtype="int16",
    device=1,
)
sd.wait()

best = 0.0
hits = 0

for i in range(0, len(audio) - 1280, 1280):
    scores = model.predict(audio[i:i + 1280, 0])
    value = float(scores.get("hey_jarvis_v0.1", 0.0))

    best = max(best, value)

    if value >= 0.5:
        hits += 1

print(f"BEST={best:.3f}")
print(f"HITS={hits}")
