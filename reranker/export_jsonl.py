import json

def export_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# example
examples = [
    {"context": "please recommend a book", "candidate": "deep learning", "label": 1},
    {"context": "please recommend a book", "candidate": "how to raise a cat", "label": 0}
]

export_jsonl(examples, 'train.jsonl')
