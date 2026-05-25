from pathlib import Path
from openai import OpenAI
import time

client = OpenAI(
    api_key="API KEY HERE"
)

INPUT_FOLDER = Path("transcripts")
OUTPUT_FOLDER = Path("improved")

OUTPUT_FOLDER.mkdir(exist_ok=True)

CHUNK_SIZE = 12000  # smaller = safer for token limits
MODEL = "gpt-5-mini"


def split_text(text, chunk_size):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # try to split at a newline, not in the middle of a sentence
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos != -1:
                end = newline_pos

        chunks.append(text[start:end].strip())
        start = end

    return [chunk for chunk in chunks if chunk]


for file in INPUT_FOLDER.glob("*.txt"):

    output_file = OUTPUT_FOLDER / file.name

    if output_file.exists():
        print(f"Skipping already improved file: {file.name}")
        continue

    print(f"Improving: {file.name}")

    text = file.read_text(encoding="utf-8")
    chunks = split_text(text, CHUNK_SIZE)

    improved_chunks = []

    for index, chunk in enumerate(chunks, start=1):

        print(f"  Processing chunk {index}/{len(chunks)}")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""
Διόρθωσε το παρακάτω ελληνικό transcript πανεπιστημιακής διάλεξης.

Κανόνες:
- Διόρθωσε λάθη απομαγνητοφώνησης.
- Βάλε σωστή στίξη.
- Βάλε παραγράφους.
- Κράτησε το ίδιο νόημα.
- Μην κάνεις περίληψη.
- Μην αφαιρέσεις πληροφορίες.
- Κάνε το κείμενο φυσικό και ακαδημαϊκό.
- Κράτησε τους χρονικούς δείκτες αν υπάρχουν.

Κείμενο:
{chunk}
"""
                }
            ]
        )

        improved_text = response.choices[0].message.content
        improved_chunks.append(improved_text)

        time.sleep(1)

    final_text = "\n\n".join(improved_chunks)

    output_file.write_text(final_text, encoding="utf-8")

    print(f"Saved: {output_file}")

print("Done.")