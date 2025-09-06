import re
import os

def second_cleanup(md_doc):
    # collapse multiple blank lines
    cleaned = re.sub(r"\n\s*\n", "\n\n", md_doc)
    cleaned = cleaned.strip()

    # phrases to exclude
    exclude_phrases = [
        "all rights reserved",
        "privacy policy",
    ]
    for phrase in exclude_phrases:
        cleaned = cleaned.replace(phrase, "")

    # remove close buttons: ×, X, [×](#)
    cleaned = re.sub(r"(\[×\]\(#\)|×|X)\s*", "", cleaned)

    # remove standalone UI junk (Ok, Yes, No, Submit)
    cleaned = re.sub(r"^(Ok|Yes|No|Submit)\s*$", "", cleaned, flags=re.MULTILINE)

    # remove entire modal/dialog sections starting with ### headers
    modal_headers = [
        "Log in to",
        "Please Login/Register",
        "Error",
        "Do you really want to logout",
        "Notification Alert",
        "Your password must contain",
        "Please Complete Your Profile",
        "Your profile is currently under moderation",
    ]
    for header in modal_headers:
        cleaned = re.sub(
            rf"### {re.escape(header)}.*?(?=(\n###|\Z))",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # 🔹 Remove stray empty bullet points like "*"
    cleaned = re.sub(r"^\*\s*$", "", cleaned, flags=re.MULTILINE)

    # final trim + collapse again after removals
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


# ✅ adjust path so it stays inside your project folder
input_folder = "data/first_clean"
out_folder = "data/second_clean"
os.makedirs(out_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if fname.endswith(".md"):
        input_path = os.path.join(input_folder, fname)
        out_path = os.path.join(out_folder, fname)
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        clean_content = second_cleanup(content)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(clean_content)
