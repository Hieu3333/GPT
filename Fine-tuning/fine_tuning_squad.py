import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Function to load and prepare the SQuAD dataset
def load_and_prepare_data():
    squad_dataset = load_dataset("rajpurkar/squad")

    def prepare_data(dataset):
        inputs = []
        labels = []
        
        for item in dataset:
            question = item['question']
            context = item['context']
            answer = item['answers']['text'][0]  # Take the first answer
            
            # Combine question and context
            input_text = f"Question: {question} Context: {context}"
            inputs.append(input_text)
            labels.append(answer)
        
        return inputs, labels

    train_inputs, train_labels = prepare_data(squad_dataset['train'])
    val_inputs, val_labels = prepare_data(squad_dataset['validation'])

    return train_inputs, train_labels, val_inputs, val_labels

# Function to tokenize the data
def tokenize_data(inputs, labels):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_encodings = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(inputs, truncation=True, padding=True, return_tensors="pt")

    class QADataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = tokenizer.encode(self.labels[idx], truncation=True, padding="max_length", max_length=20)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = QADataset(train_encodings, train_labels)
    val_dataset = QADataset(val_encodings, val_labels)

    return train_dataset, val_dataset

# Function to set up and train the model
def train_model(train_dataset, val_dataset):
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir='./results',              # Output directory
        num_train_epochs=3,                  # Number of training epochs
        per_device_train_batch_size=4,       # Batch size for training
        per_device_eval_batch_size=4,        # Batch size for evaluation
        warmup_steps=500,                     # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,                   # Strength of weight decay
        logging_dir='./logs',                 # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",          # Evaluate at the end of each epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("fine-tuned-gpt2-squad")
    tokenizer.save_pretrained("fine-tuned-gpt2-squad")

# Main execution flow
def main():
    train_inputs, train_labels, val_inputs, val_labels = load_and_prepare_data()
    train_dataset, val_dataset = tokenize_data(train_inputs + val_inputs, train_labels + val_labels)
    train_model(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
