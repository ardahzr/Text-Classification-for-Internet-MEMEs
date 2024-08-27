import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss

df = pd.read_csv('/home/libuntu/Desktop/MEME/archive/MIMIC2024.csv')

df = df[['ExtractedText', 'Misogyny', 'Objectification', 'Prejudice', 'Humiliation']]

label_columns = ['Misogyny', 'Objectification', 'Prejudice', 'Humiliation']

df['ExtractedText'].fillna('', inplace=True)  

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

for label_col in label_columns:
    print(f"---- {label_col} etiketi için eğitim ----")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df[['ExtractedText', label_col]])
    test_dataset = Dataset.from_pandas(test_df[['ExtractedText', label_col]])


    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


    def preprocess_function(examples):
    
        inputs = tokenizer(examples['ExtractedText'], padding='max_length', truncation=True)
    
        inputs['labels'] = examples[label_col]
        return inputs

  
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

   
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

    
    training_args = TrainingArguments(
        output_dir=f'./results/{label_col}',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs/{label_col}',
        logging_steps=10,
        evaluation_strategy="epoch"
    )


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset
    )


    trainer.train()

    results = trainer.evaluate()
    print(results)


    model.save_pretrained(f'./saved_model/{label_col}')
    tokenizer.save_pretrained(f'./saved_model/{label_col}')
