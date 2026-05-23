
import json
import subprocess
import os
import sys

def process_batch(batch_items):
    prompt = """You are the 'Cognitive Refiner'. Analyze the cause-and-effect flow: User(t-1) -> AI(t) -> User(t).
Extract insights about the user's preferences, values, or decision-making style.

Input Data:
{data}

Output Format:
JSON array of SeedItem objects:
- content (str): [Topic / Action] The extracted preference/value. Embed the topic and action context directly at the beginning of the text (e.g., "[News Report / Action: Adjust narrative flow] Aims to make sources less specific...").
- type (str): 'decision', 'preference', 'value', or 'anchor_candidate'
- context (dict): Contains ONLY 'domain_tags'. Do NOT include 'action' or 'topic' fields here. (e.g., {{"domain_tags": ["news-reporting"]}})
- source_uuid (str): The UUID of the conversation

Respond ONLY with the JSON array. No other text.
""".format(data=json.dumps(batch_items, indent=2))

    try:
        # Run gemini -p with the prompt
        result = subprocess.run(['gemini', '-p', prompt], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running gemini: {result.stderr}", file=sys.stderr)
            return None
        
        # Extract JSON from the output (it might contain markdown blocks)
        output = result.stdout.strip()
        if output.startswith("```json"):
            output = output[7:]
        if output.endswith("```"):
            output = output[:-3]
        
        try:
            return json.loads(output.strip())
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from output: {output[:100]}...", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Exception during batch processing: {e}", file=sys.stderr)
        return None

def main():
    start_index = 0
    batch_size = 20
    input_file = 'extracted_pairs.json'
    output_file = 'final_seeds.json'

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        all_data = json.load(f)

    # Initialize final_seeds.json if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f)

    total_items = len(all_data)
    print(f"Starting processing from index {start_index} to {total_items}")

    for i in range(start_index, total_items, batch_size):
        end_idx = min(i + batch_size, total_items)
        batch = all_data[i:end_idx]
        print(f"Processing batch {i} to {end_idx}...")
        
        results = process_batch(batch)
        if results:
            with open(output_file, 'r+') as f:
                current_seeds = json.load(f)
                current_seeds.extend(results)
                f.seek(0)
                json.dump(current_seeds, f, indent=2)
                f.truncate()
            print(f"Successfully processed {len(results)} insights from batch.")
        else:
            print(f"Skipping batch {i} due to errors.")
        
        # Stop if we hit a certain limit to avoid taking too many turns in one go if this were interactive
        # But here I'll just run it.

if __name__ == "__main__":
    main()
