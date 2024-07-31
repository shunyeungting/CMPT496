import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data  #Import preprocess function
import os

# Set training parameters
epochs = 4  
learning_rate = 5e-5  
batch_size = 32  
validation_batch_size = 32  
test_batch_size = 32  

def evaluate(model, dataloader, criterion):
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    true_labels = []

    # Disable gradient calculation
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Compute loss
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            # Convert logits to probabilities and extract predicted labels
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
            predictions.extend(predicted_labels.tolist())
            true_labels.extend(labels.tolist())

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    average_loss = total_loss / total_samples
    return average_loss, accuracy

def train_model(dataset_name, model_name):
    # Load preprocessed data
    train_inputs, train_masks, train_labels, test_inputs, test_masks, test_labels = preprocess_data(dataset_name, model_name)
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
    
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Define DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)
    
    # Train the model
    for epoch in range(epochs):  #  Train epoch
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Compute loss
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validate the model
        val_loss, val_accuracy = evaluate(model, test_dataloader, criterion)
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion)
    print(f"Testing Loss: {test_loss}, Testing Accuracy: {test_accuracy}")
    
    # Check and create directory
    model_dir = "TrainedPunModel"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    print("Model training completed and saved.")


if __name__ == "__main__":
    dataset_name = 'frostymelonade/SemEval2017-task7-pun-detection'  
    model_name = 'frostymelonade/roberta-small-pun-detector-v2'  
    train_model(dataset_name, model_name)