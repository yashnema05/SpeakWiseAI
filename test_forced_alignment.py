
from modules.forced_alignment import forced_align

audio_path = input(
    "Apni audio recording ka full path paste karo: "
).strip().strip('"')

transcript = input(
    "Recording mein jo bola hai, uska exact transcript likho: "
).strip()

try:
    results = forced_align(audio_path, transcript)

    print("\n--- WORD ALIGNMENT RESULTS ---")

    for item in results:
        print(
            f"Word: {item['word']:<20} "
            f"Start: {item['start']} sec | "
            f"End: {item['end']} sec | "
            f"Score: {item['score']}"
        )

except Exception as exc:
    print(f"\nAlignment failed: {exc}")
