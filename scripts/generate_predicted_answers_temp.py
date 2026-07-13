import json
import subprocess

# Path to your input questions file
INPUT_FILE = "questions.json"

# Read questions
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    questions = json.load(f)

# Temperatures to test
temperatures = [0.3, 0.7, 1.0]

for temp in temperatures:
    print(f"\n=== Generating answers with temperature {temp} ===")
    results = []

    for q in questions:
        prompt = q["question"]

        # Run Ollama model with temperature parameter
        command = [
            "ollama", "run", "meta-llama-3-8b-instruct.Q4_K_M.gguf",
            "--temperature", str(temp)
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt)

        if error:
            print(f"Error for question: {prompt[:30]}... -> {error}")

        results.append({
            "question": prompt,
            "answer": output.strip(),
            "temperature": temp
        })

    # Save each version separately
    out_file = f"predicted_answers_temp{temp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved answers to {out_file}")
