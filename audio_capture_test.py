import time

from core.audio_engine import (
    CommandCapture,
    MicrophoneStream,
    NoiseFloorTracker,
)


def main():
    mic = MicrophoneStream(device=1)
    noise = NoiseFloorTracker()
    capture = CommandCapture(mic, noise)

    mic.start()

    try:
        print("Calibrating... stay completely silent for 3 seconds.")

        end_time = time.time() + 3

        while time.time() < end_time:
            chunk = mic.read_chunk(timeout=0.2)

            if chunk is not None:
                noise.update(chunk)

        print(f"Calibration complete. Noise floor: {noise.floor:.1f}")
        print("NOW SAY: NEXUS, THIS IS A TEST")

        audio = capture.capture()

        if audio is None:
            print("RESULT: No speech captured.")
        else:
            print(f"RESULT: Captured {len(audio)} samples.")
            print(f"RESULT: Duration {len(audio) / 16000:.2f} seconds.")

    finally:
        mic.stop()


if __name__ == "__main__":
    main()