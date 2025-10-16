import json
import time
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.evaluate import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.models.llms.ollama_model import OllamaModel

# -----------------------------
# Latency & Token Metric
# -----------------------------
class LatencyAndTokenMetric:
    def __init__(self, model):
        self.model = model
        self.name = "LatencyAndTokenEfficiency"

    def score(self, query, answer, context):
        start_time = time.time()
        ctx = ""
        if isinstance(context, list):
            ctx = "\n\n".join(str(c) for c in context)
        input_tokens = len(query.split()) + len(ctx.split())
        output_tokens = len(answer.split())
        elapsed = time.time() - start_time

        return {
            "latency_seconds": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_total": input_tokens + output_tokens,
            "efficiency_ratio": output_tokens / (input_tokens + 1e-6)
        }

# -----------------------------
# Load predicted answers JSON
# -----------------------------
file_path = "predicted_answers.json"  # update path if needed
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# -----------------------------
# Create Goldens
# -----------------------------
goldens = []
for item in data:
    context_texts = []
    for chunk in item.get("Retrieved_Chunks", []):
        if isinstance(chunk, dict) and "chunk_text" in chunk:
            # truncate long chunks to avoid prompt overflow
            context_texts.append(" ".join(chunk["chunk_text"].split()[:300]))
        elif isinstance(chunk, str):
            context_texts.append(" ".join(chunk.split()[:300]))
    goldens.append(
        Golden(
            input=item["Question"],
            actual_output=item["Predicted_Answer"],
            expected_output=item["Answer"],
            retrieval_context=context_texts
        )
    )

dataset = EvaluationDataset(goldens=goldens)

# -----------------------------
# Convert Goldens to Test Cases
# -----------------------------
test_cases = []
for g in dataset.goldens:
    test_cases.append(
        LLMTestCase(
            input=g.input,
            actual_output=g.actual_output,
            expected_output=g.expected_output,
            retrieval_context=g.retrieval_context
        )
    )

# -----------------------------
# Initialize OllamaModel
# -----------------------------
ollama_model = OllamaModel(model="smollm:135m")  # lightweight for laptop

# -----------------------------
# Faithfulness Metric (sync)
# -----------------------------
metric = FaithfulnessMetric(
    model=ollama_model,
    include_reason=True,
    async_mode=False  # run synchronously to avoid JSON truncation
)

# -----------------------------
# Run evaluation
# -----------------------------
eval_results = evaluate(test_cases=test_cases, metrics=[metric])

# -----------------------------
# Compute latency & token usage
# -----------------------------
latency_metric = LatencyAndTokenMetric(model=ollama_model)
latency_results = []
for g in dataset.goldens:
    latency_results.append(latency_metric.score(
        query=g.input,
        answer=g.actual_output,
        context=g.retrieval_context
    ))

# -----------------------------
# Print results
# -----------------------------
print("\n✅ Faithfulness Evaluation Complete!\n")
faithfulness_scores = []
for i, testRes in enumerate(eval_results.test_results, 1):
    print(f"🧪 Test Case {i}: {testRes.input[:80]}...")
    score_dict = {}
    for m in testRes.metrics_data:
        print(f"   {m.name}: {m.score:.2f}")
        score_dict[m.name] = m.score
    faithfulness_scores.append({
        "question": testRes.input,
        "faithfulness_score": score_dict,
        "actual_answer": testRes.actual_output,
        "expected_answer": testRes.expected_output
    })

print("\n⏱️ Latency & Token Efficiency:")
for i, res in enumerate(latency_results, 1):
    print(f"Q{i}: Latency={res['latency_seconds']:.3f}s, Tokens={res['tokens_total']}, Efficiency={res['efficiency_ratio']:.2f}")

# -----------------------------
# Save results to JSON
# -----------------------------
output_file = "faithfulness_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(faithfulness_scores, f, ensure_ascii=False, indent=4)

print(f"\n✅ Faithfulness scores saved to {output_file}")
