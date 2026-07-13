import sys
from pathlib import Path
import json

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.transformer.decision_engine import DecisionEngine
from src.utils.constants import OUTPUT_DIR

# Define a config using Qwen/Qwen2.5-0.5B-Instruct which is a valid instruction-following model
config = {
    "decision_engine": {
        "high_confidence_threshold": 0.90,
        "low_confidence_threshold": 0.60
    },
    "llm": {
        "enabled": True,
        "provider": "huggingface",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_new_tokens": 128,
        "temperature": 0.2
    }
}

model_dir = OUTPUT_DIR / "models" / "best_model"
retriever_index_dir = OUTPUT_DIR / "retrieval_index"

print("Initializing DecisionEngine with Qwen/Qwen2.5-0.5B-Instruct...")
engine = DecisionEngine(
    model_dir=model_dir,
    retriever_index_dir=retriever_index_dir,
    config=config
)

print("Routing ticket...")
res = engine.route_ticket("reset card pin number")

print("Result:")
print(json.dumps(res, indent=2))

# Save output to scratch/outputs/debug_output.json
output_dir = REPO_ROOT / "scratch" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "debug_output.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"Output saved to: {output_dir / 'debug_output.json'}")
