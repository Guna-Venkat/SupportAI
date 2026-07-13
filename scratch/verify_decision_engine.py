import os
import sys
import json
from pathlib import Path

# Ensure repo root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set environment variables for testing
os.environ["TESTING"] = "true"

from src.models.transformer.decision_engine import DecisionEngine
from src.utils.constants import OUTPUT_DIR

# Define a custom config to test fallback and escalation paths
config = {
    "decision_engine": {
        "high_confidence_threshold": 0.90,
        "low_confidence_threshold": 0.30
    },
    "llm": {
        "enabled": True,
        "provider": "huggingface",
        "model_id": "hf-internal-testing/tiny-random-gpt2",
        "max_new_tokens": 30,
        "temperature": 0.0
    }
}

model_dir = OUTPUT_DIR / "models" / "best_model"
retriever_index_dir = OUTPUT_DIR / "retrieval_index"

print("Initializing DecisionEngine...")
engine = DecisionEngine(
    model_dir=model_dir,
    retriever_index_dir=retriever_index_dir,
    config=config
)

results = {}

# 1. Tier 1: High Confidence
print("\n--- Testing Tier 1: High Confidence ---")
res_high = engine.route_ticket("I forgot my passcode and cannot login")
print(json.dumps(res_high, indent=2))
results["high_confidence"] = res_high

# 2. Tier 2: Mid Confidence -> Fallback
print("\n--- Testing Tier 2: Mid Confidence -> LLM Fallback ---")
res_mid = engine.route_ticket("reset card pin number")
print(json.dumps(res_mid, indent=2))
results["mid_confidence_fallback"] = res_mid

# 3. Tier 3: Low Confidence -> Escalation
print("\n--- Testing Tier 3: Low Confidence -> Human Escalation ---")
res_low = engine.route_ticket("xyz abc qwe rty")
print(json.dumps(res_low, indent=2))
results["low_confidence_escalation"] = res_low

# Save output
output_dir = REPO_ROOT / "scratch" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "decision_engine_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nAll tests completed. Results saved to: {output_dir / 'decision_engine_results.json'}")
