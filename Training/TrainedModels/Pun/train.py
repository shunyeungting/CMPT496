import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data  # Import preprocess function
import os
import time
from datasets import load_dataset
from tqdm import tqdm 

print("Current working directory:", os.getcwd())

# Set training parameters
epochs = 4  
learning_rate = 5e-5  
batch_size = 32  
validation_batch_size = 32  
test_batch_size = 32  

def get_device():
    """Returns the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        return torch.device('mps')
    else:
        print("Neither CUDA nor MPS is available. Using CPU.")
        return torch.device('cpu')

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
            predictions.extend(predicted_labels.tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    average_loss = total_loss / total_samples
    return average_loss, accuracy

def train_model(dataset_name, model_name, resume_training=False, model_path=None):
    device = get_device()
    print(f"Using device: {device}")

    train_inputs, train_masks, train_labels, test_inputs, test_masks, test_labels = preprocess_data(dataset_name, model_name)
    
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if resume_training and model_path:
        model.load_state_dict(torch.load(model_path))

    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    for epoch in range(epochs):  
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")  # adding progress bar
        for batch in progress_bar:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # progress bar update
            progress_bar.set_postfix(loss=loss.item())

        val_loss, val_accuracy = evaluate(model, test_dataloader, criterion, device)
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    
    model.eval()
    with torch.no_grad():
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)
    print(f"Testing Loss: {test_loss}, Testing Accuracy: {test_accuracy}")
    
    model_dir = "TrainedPunModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"pytorch_model_{timestamp}.bin"
    
    torch.save(model.state_dict(), os.path.join(model_dir, model_filename))
    print("Model training completed and saved as", model_filename)

def predict_with_saved_model(model_name, model_path, dataset_name):
    # check device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load saved model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # adding weights_only=True
    model.to(device)
    model.eval()

    # load now dataset
    dataset = load_dataset(dataset_name, split='train')

    # predict
    texts = [item['text'] for item in dataset]  # assume there is "text"
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    # result
    for text, pred in zip(texts[:5], predictions[:5]):
        print(f"Text: {text}\nPrediction: {'Offensive' if pred == 1 else 'Not Offensive'}\n")

if __name__ == "__main__":
    dataset_name = 'frostymelonade/SemEval2017-task7-pun-detection'
    model_name = 'frostymelonade/roberta-small-pun-detector-v2'
    
    # ask if user want to resume or start over
    resume_training = input("Do you have a pre-trained model to resume training? (Y/N): ").strip().upper()
    model_path = None
    if resume_training == 'Y':
        model_path = input("Please provide the path to the pre-trained model: ").strip()
        if not os.path.exists(model_path):
            print("Model path does not exist. Exiting.")
            exit()

        # load saved model and predict
        predict_with_saved_model(model_name, model_path, 'metaeval/offensive-humor')
    else:
        train_model(dataset_name, model_name, resume_training == 'Y', model_path)


