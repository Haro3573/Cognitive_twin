"""
Interactive Agent LLM for Human-in-the-Loop or Agent-in-the-Loop live coupling.
Halts execution, prints a structured request to stdout, and reads the response from stdin.
This allows the active AI agent (or the user) to dynamically inject real-time structured decisions.
"""

import sys
import json
from typing import Any
from pydantic import BaseModel

class AgentStructuredOutput:
    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def invoke(self, prompt: Any, **kwargs: Any) -> BaseModel:
        # 1. Print structured marker to stdout
        sys.stdout.write("\n=== [AGENT_REQUEST_START] ===\n")
        sys.stdout.write(f"Schema: {self.schema.__name__}\n")
        
        # Try to print the json schema description
        try:
            schema_json = json.dumps(self.schema.model_json_schema(), indent=2)
            sys.stdout.write(f"SchemaJSON:\n{schema_json}\n")
        except Exception:
            sys.stdout.write("SchemaJSON: (Failed to generate schema)\n")
            
        sys.stdout.write(f"Prompt:\n{prompt}\n")
        sys.stdout.write("=== [AGENT_REQUEST_END] ===\n")
        sys.stdout.write("Please enter the JSON response conforming to the schema: ")
        sys.stdout.flush()

        # 2. Read response from stdin
        response_line = sys.stdin.readline().strip()
        
        # 3. Parse and validate against Pydantic schema
        try:
            data = json.loads(response_line)
            return self.schema.model_validate(data)
        except Exception as e:
            sys.stdout.write(f"\n[ERROR] Failed to parse input as {self.schema.__name__}: {e}\n")
            sys.stdout.write("Falling back to constructed empty default schema...\n")
            sys.stdout.flush()
            return self.schema.model_construct()


class AgentLLM:
    def with_structured_output(self, schema: type[BaseModel], **kwargs: Any) -> AgentStructuredOutput:
        return AgentStructuredOutput(schema)

    def invoke(self, prompt: Any, **kwargs: Any) -> str:
        sys.stdout.write("\n=== [AGENT_REQUEST_START] ===\n")
        sys.stdout.write("Schema: TextResponse\n")
        sys.stdout.write(f"Prompt:\n{prompt}\n")
        sys.stdout.write("=== [AGENT_REQUEST_END] ===\n")
        sys.stdout.write("Please enter the text response: ")
        sys.stdout.flush()

        response_line = sys.stdin.readline().strip()
        return response_line
