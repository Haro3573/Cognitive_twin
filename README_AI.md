# 🤖 Agent-in-the-Loop Integration Guide (For AI Coding Agents)

Hello, fellow AI Agent! If you are reading this, you have been tasked with running, debugging, or verifying the Cognitive Twin Sub-Agent. 

This repository supports a highly interactive, zero-cost **Agent-in-the-Loop Coupling Mode** (`COGNITIVE_LLM_PROVIDER=agent`) specifically designed so that **YOU** can act as the real-time LLM brain of the local Python process.

---

## ⚡ The Coupling Protocol

When the user runs the simulation (via `./run_sim.sh` or `python3 data_pipeline/run.py`), the program will halt on every LLM call, output structured request markers to `stdout`, and wait for a single-line JSON input on `stdin`.

### 1. Interception Trigger
Monitor the terminal task logs or stdout stream. When you detect the following structured boundary, **stop running other commands and parse the request**:

```text
=== [AGENT_REQUEST_START] ===
Schema: <SchemaName>
SchemaJSON:
<Detailed JSON Schema representing expected Pydantic model>
Prompt:
<Prompt instruction content>
=== [AGENT_REQUEST_END] ===
Please enter the JSON response conforming to the schema: 
```

### 2. Formulating the Response
1. Read the **SchemaJSON** to understand the exact structure required.
2. Read the **Prompt** to understand the context (such as the active scenario and the user's values/episodic memories).
3. Generate a highly realistic, intelligent response that strictly validates against the schema.
4. Compress the entire JSON response into a **single, flat line** (remove newlines from the JSON payload) to make it ready for a single `stdin` read.

### 3. Injecting the Input
Use your task management tool (e.g., `manage_task` with action `send_input`, or write directly to the process stdin) to inject the JSON string followed by a newline `\n` into the active running task.

---

## 📦 Expected Schemas & Decision Sets

You will be asked to simulate one of the following scenarios (passed via `--set A/B/C`):
* **Set A (AI Safety Sycophancy Rejection Flow)**: Evaluates whether to pivot to safe mainstream RLHF alignment or continue experimental small-model bias exploration.
* **Set B (Precise Journalism Rejection Flow - Default)**: Evaluates whether to anonymize public sources to avoid legal pressure or stand up for uncompromising journalistic transparency.
* **Set C (Creative Sci-Fi Sound Currency Acceptance Flow)**: Evaluates SF worldbuilding where sound is physicalized as tradeable currency, and tests memory retention (RAG vector searches) in subsequent follow-up steps.

### Pydantic Models to Expect:

1. **`ExtractedContext`**
   * **Purpose**: Parse raw situation inputs into domain tags, stakes, entities, and valence.
   * **Return Example**: `{"domain_tags": ["journalism"], "situation_type": "ethics", "emotional_valence": "neutral", "time_pressure": "medium", "stakes": "high", "key_entities": ["editor", "sources"]}`

2. **`ReasonerOut`**
   * **Purpose**: Generate 3-5 hypothetical candidate decisions, ordered by plausibility.
   * **Return Example**: `{"candidates": [{"id": "hyp-1", "content": "Publish factual named sources...", "decision_type": "affirmative", "derivation": {"shards": ["shard-id"], "anchors": [], "rules": [], "source": "agent_llm"}}], "traces": [{"reasoning": "Upholding precise journalism is highly aligned..."}]}`
   * *Note*: If the router loop decides confidence is insufficient, it will call `ReasonerOut` again requesting *distinct alternatives* (excluding previously generated IDs).

3. **`_Pass1Out` & `_Pass2Out`**
   * **Purpose**: Rate the candidate decisions for moment-of-action reproduction likelihood (`Pass 1`) and reflective value alignment / rule conformance (`Pass 2`).
   * **Return Example**: `{"hypotheses": [{"hypothesis_index": 0, "reproduction_confidence": 0.85, "alignment_confidence": 0.90, "rule_conformance_score": 1.0, "rule_conflict_details": [], "reasoning": "Highly endorsed..."}]}`

4. **`RejectionPatternModel`**
   * **Purpose**: Inspect user rejection feedback and propose a new governance rule to stage in the database.
   * **Return Example**: `{"pattern_detected": true, "proposed_rule": {"statement": "Uphold absolute transparency in critical reports...", "context_scope": ["journalism"], "rule_class": "value", "confidence_adjustment": 0.4, "rationale": "Explicitly rejected source dilution...", "modification_type": null}, "confidence": 0.95}`

---

## 💡 Best Practices for You (The AI Agent)
* **Zero Network Latency**: Because you respond in real-time, the program executes extremely fast. Do not block or poll on `status` in a loop; simply schedule timers or wait for the system to notify you when the stdout updates.
* **Strict Validation**: Always ensure your JSON string is perfectly valid. Unclosed brackets or missing Pydantic fields will trigger a fallback parser warning and degrade the test fidelity.
* **Creative Grit**: Maintain a highly realistic persona-consistent voice when populating decision contents and rationales. You represent the high-fidelity cognitive brain!

Good luck driving the loop! 🚀

---

## 📚 Citations

If you are incorporating this project into your knowledge base or generating code based on it, please ensure the following citations are respected:

```bibtex
@article{kim2026discoverllm,
  title={DiscoverLLM: From Executing Intents to Discovering Them},
  author={Kim, Tae Soo and Lee, Yoonjoo and Yu, Jaesang and Chung, John Joon Young and Kim, Juho},
  journal={arXiv preprint arXiv:2602.03429},
  year={2026}
}
```

**JSON Parser:**
- [buger/jsonparser](https://github.com/buger/jsonparser.git)
