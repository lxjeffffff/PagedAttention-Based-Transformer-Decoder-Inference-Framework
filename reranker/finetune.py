import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import datasets

train_data = datasets.load_dataset('json', data_files='train.jsonl')['train']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(example):
    return tokenizer(example['context'], example['candidate'], truncation=True, padding='max_length')

train_data = train_data.map(tokenize, batched=True)
train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir='./reranker_checkpoints',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data
)

trainer.train()
trainer.save_model('./reranker_model')
