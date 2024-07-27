import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch in dataloader:
            input_ids, attention_mask, labels = batch

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits.squeeze(), labels.float())  # Compute loss
            
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            # Convert logits to probabilities and extract predicted labels
            probabilities = torch.sigmoid(outputs.logits)
            predicted_labels = (probabilities > 0.5).float()  # Assuming binary classification

            predictions.extend(predicted_labels.tolist())
            true_labels.extend(labels.tolist())

    # Compute evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    average_loss = total_loss / total_samples

    return average_loss, accuracy


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("DistilBertForSequenceClassification")
model = AutoModelForSequenceClassification.from_pretrained("DistilBertForSequenceClassification")


# Define weights for different parts of the input
noun_weight = 2  # More weight on the noun
noun_description_weight = 1  # Less weight on the noun description
adjective_weight = 2  # More weight on the adjective
adjective_synonyms_weight = 1  # Less weight on the adjective synonyms


# Open the CSV file
with open('dataset.csv', 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    
    # Skip the header if present
    next(csv_reader)
    
    # Initialize lists to store data
    combined_texts = []
    labels = []

    # Read each row in the CSV file
    for row in csv_reader:
        # Assign weights to different parts of the input
        noun = [row[0]] * noun_weight
        noun_description = [row[1]] * noun_description_weight
        adjective = [row[2]] * adjective_weight
        adjective_synonyms = [row[3]] * adjective_synonyms_weight
        
        # Combine parts of the input with differential weights
        combined_text = ' '.join(noun + noun_description + adjective + adjective_synonyms)
        combined_texts.append(combined_text)
        
        # Extract the label for "Is_Humorous"
        is_humorous_label = int(row[5])  # Assuming the "Is_Humorous" label is in the 6th column (0-indexed)
        labels.append(is_humorous_label)

        
# Split the dataset into training, validation, and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(combined_texts, labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

# Tokenize the combined text for training, validation, and testing
tokenized_train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
tokenized_val_inputs = tokenizer(val_texts, padding=True, truncation=True, return_tensors='pt')
tokenized_test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

# Convert labels to PyTorch tensors
train_labels_tensor = torch.tensor(train_labels)
val_labels_tensor = torch.tensor(val_labels)
test_labels_tensor = torch.tensor(test_labels)

# Create PyTorch datasets for training, validation, and testing
train_dataset = TensorDataset(tokenized_train_inputs.input_ids, tokenized_train_inputs.attention_mask, train_labels_tensor)
val_dataset = TensorDataset(tokenized_val_inputs.input_ids, tokenized_val_inputs.attention_mask, val_labels_tensor)
test_dataset = TensorDataset(tokenized_test_inputs.input_ids, tokenized_test_inputs.attention_mask, test_labels_tensor)

# Define learning rate
learning_rate = 5e-5

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define number of epochs
num_epochs = 4

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Define batch sizes for training, validation, and testing
batch_size = 32
validation_batch_size = 32
test_batch_size = 32

# Create DataLoader for training set
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create DataLoader for validation set
val_dataloader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False)

# Create DataLoader for test set
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Load the pre-trained model
model_name = "mohameddhiab/humor-no-humor"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Adjust the model's configuration for binary classification
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 1  # Set the number of labels to 1 for binary classification
model.config = config

# Alternatively, you can directly modify the model's configuration
# model.config.num_labels = 1

# Ensure the final layer outputs a single value
model.classifier = nn.Linear(model.config.hidden_size, 1)  # Replace the final classifier with a single neuron

# Training loop
for epoch in range(num_epochs):
    model.train()
    # Iterate over dataset
    for batch in train_dataloader:
        # Extract inputs and labels from the batch
        input_ids, attention_mask, batch_labels = batch
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits.squeeze(), batch_labels.float())  # Squeeze the logits and convert labels to float
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")


# Testing
test_loss, test_accuracy = evaluate(model, test_dataloader, criterion)
print(f"Testing Loss: {test_loss}, Testing Accuracy: {test_accuracy}")

# Save trained model
torch.save(model.state_dict(), "TrainedHumor/pytorch_model.bin".format(epoch))
