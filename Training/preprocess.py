import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def preprocess_data(dataset_name, model_name):
    # Load the dataset from hugging face
    dataset = load_dataset(dataset_name)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize and encode the text
    def tokenize_data(data):
        return tokenizer(data['text'], padding=True, truncation=True, return_tensors="pt")
    
    # Tokenize and encode the dataset
    inputs = tokenize_data(dataset['test'])
    labels = torch.tensor(dataset['test']['label'])
    
    # Split the data into training and testing sets (90% train, 10% test)
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs.input_ids, labels, test_size=0.1, random_state=42, stratify=labels)
    
    train_masks, test_masks, _, _ = train_test_split(
        inputs.attention_mask, labels, test_size=0.1, random_state=42, stratify=labels)
    
    return train_inputs, train_masks, train_labels, test_inputs, test_masks, test_labels

if __name__ == "__main__":
    dataset_name = 'frostymelonade/SemEval2017-task7-pun-detection'  
    model_name = 'frostymelonade/roberta-small-pun-detector-v2'  
    train_inputs, train_masks, train_labels, test_inputs, test_masks, test_labels = preprocess_data(dataset_name, model_name)
    print("Data preprocessing completed.")