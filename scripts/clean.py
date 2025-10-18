import re
import os

def safe_cleanup(md_doc):
    # Normalize line endings
    cleaned = re.sub(r'\r\n?', '\n', md_doc)

    # Remove repeated blank lines (2+ → 2)
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    # List of phrases to remove (case-insensitive)
    exclude_phrases = [
        "all rights reserved",
        "privacy policy",
        "SISFSYou need to enable JavaScript to run this app.",
        "Loading..."
    ]
    for phrase in exclude_phrases:
        cleaned = re.sub(re.escape(phrase), '', cleaned, flags=re.IGNORECASE)

    # Split into lines and remove lines that are truly empty
    lines = cleaned.split('\n')
    lines = [line for line in lines if line.strip() != '']

    # Join lines back
    cleaned = '\n'.join(lines)

    return cleaned

# --- Process all markdown files ---
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

        clean_content = safe_cleanup(content)

        # Debug: check first few lines
        print(f"--- {fname} ---")
        print("\n".join(clean_content.splitlines()[:10]))
        print("-----------------")

        with open(out_path, 'w', encoding="utf-8") as f:
            f.write(clean_content)

        print(f"✅ Cleaned {fname}")
