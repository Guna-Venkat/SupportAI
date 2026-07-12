"""
test_decision_engine.py
=======================
Unit and integration tests for Phase 12 Decision Engine.
"""

from src.models.transformer.decision_engine import DecisionEngine
from src.utils.constants import OUTPUT_DIR


def test_decision_engine_routing() -> None:
    model_dir = OUTPUT_DIR / "models" / "best_model"
    retriever_index_dir = OUTPUT_DIR / "retrieval_index"

    # Define low-resource config using tiny-random-gpt2 model
    config = {
        "decision_engine": {
            "high_confidence_threshold": 0.90,
            "low_confidence_threshold": 0.60,
        },
        "llm": {
            "enabled": True,
            "provider": "huggingface",
            "model_id": "hf-internal-testing/tiny-random-gpt2",
            "max_new_tokens": 10,
            "temperature": 0.0,
        },
    }

    # Initialize decision engine
    engine = DecisionEngine(
        model_dir=model_dir,
        retriever_index_dir=retriever_index_dir,
        config=config,
    )

    # 1. Test High Confidence routing (Auto Route)
    # Let's override threshold temporarily to force high confidence route
    engine.high_threshold = 0.01
    res_high = engine.route_ticket("I want to reset my passcode")
    assert res_high["route"] == "classifier"
    assert res_high["llm_used"] is False
    assert len(res_high["retrieved_docs"]) == 0
    assert "intent" in res_high
    assert "confidence" in res_high

    # 2. Test Mid Confidence fallback routing (should trigger LLM generation)
    engine.high_threshold = 0.99
    engine.low_threshold = 0.01
    res_mid = engine.route_ticket("passcode problem")
    assert res_mid["route"] == "fallback"
    assert res_mid["llm_used"] is True
    assert len(res_mid["retrieved_docs"]) == 3
    assert "reply" in res_mid
    assert isinstance(res_mid["reply"], str)

    # 3. Test Low Confidence human escalation routing
    engine.high_threshold = 0.99
    engine.low_threshold = 0.95
    res_low = engine.route_ticket("passcode problem")
    assert res_low["route"] == "human_escalation"
    assert res_low["llm_used"] is False
    assert len(res_low["retrieved_docs"]) == 0
    assert res_low["intent"] == "unknown"
