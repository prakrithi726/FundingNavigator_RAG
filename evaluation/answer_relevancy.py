import json
import time
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import LLMTestCase

from deepeval.evaluate.evaluate import evaluate
from deepeval.metrics import AnswerRelevancyMetric
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
            context_texts.append(chunk["chunk_text"])
        elif isinstance(chunk, str):
            context_texts.append(chunk)
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
# Answer Relevancy Metric (sync)
# -----------------------------
metric = AnswerRelevancyMetric(
    model=ollama_model,
    include_reason=False,
    async_mode=False  # force synchronous evaluation
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
print("\n✅ Answer Relevancy Evaluation Complete!\n")
for i, testRes in enumerate(eval_results.test_results, 1):
    print(f"🧪 Test Case {i}: {testRes.input[:80]}...")
    for m in testRes.metrics_data:
        print(f"   {m.name}: {m.score:.2f}")

print("\n⏱️ Latency & Token Efficiency:")
for i, res in enumerate(latency_results, 1):
    print(f"Q{i}: Latency={res['latency_seconds']:.3f}s, Tokens={res['tokens_total']}, Efficiency={res['efficiency_ratio']:.2f}")

# -----------------------------
# Save full results to JSON for GitHub
# -----------------------------
results_to_save = []
for i, testRes in enumerate(eval_results.test_results):
    latency_info = latency_results[i] if i < len(latency_results) else {}
    results_to_save.append({
        "Question": testRes.input,
        "Predicted_Answer": testRes.actual_output,
        "Actual_Answer": testRes.expected_output,
        "Answer_Relevancy": testRes.metrics_data[0].score if testRes.metrics_data else None,
        "Latency_Seconds": latency_info.get("latency_seconds"),
        "Input_Tokens": latency_info.get("input_tokens"),
        "Output_Tokens": latency_info.get("output_tokens"),
        "Total_Tokens": latency_info.get("tokens_total"),
        "Efficiency_Ratio": latency_info.get("efficiency_ratio")
    })

with open("answer_relevancy_full_results.json", "w", encoding="utf-8") as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=2)

print("\n✅ Full evaluation results saved to 'answer_relevancy_full_results.json'")
