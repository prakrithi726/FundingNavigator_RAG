import re
import os

def second_cleanup(md_doc):
    cleaned = re.sub(r"\n\s*\n", '\n\n', md_doc)
    cleaned = cleaned.strip()
    exclude_phrases = [
        "all rights reserved",
        " privacy policy",
        "SISFSYou need to enable JavaScript to run this app.",
        "Loading..."
    ]
    for phrase in exclude_phrases:
        cleaned = cleaned.replace(phrase, '')
    return cleaned

input_folder = 'data'
out_folder = "data/cleaned"
os.makedirs(out_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.endswith(".md"):
        input_path = os.path.join(input_folder, fname)
        out_path = os.path.join(out_folder, fname)

        # Skip if already cleaned
        if os.path.exists(out_path):
            print(f"⏩ Skipping {fname}, already cleaned")
            continue

        with open(input_path, 'r', encoding="utf-8") as f:
            content = f.read()

        clean_content = second_cleanup(content)

        with open(out_path, 'w', encoding="utf-8") as f:
            f.write(clean_content)

        print(f"✅ Cleaned {fname}")
