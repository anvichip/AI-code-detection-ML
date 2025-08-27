import json

def load_jsonl(filepath, label):
    with open(filepath, 'r') as f:
        return [(json.loads(line).get('code', ''), label) for line in f]
