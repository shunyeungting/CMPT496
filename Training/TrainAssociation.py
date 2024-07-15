import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

"""Method for training association with SentenceTransformers from https://www.sbert.net/docs/training/overview.html"""

# Load dataset
df = pd.read_csv('dataset.csv')

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Define a list to hold the formatted examples
train_examples = []

# Define the weight for repeating the noun and adjective
noun_repeats = 3 
adjective_repeats = 3

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # Concatenate Noun multiple times
    repeated_noun = (row['Noun'] + ' ') * noun_repeats
    
    # Concatenate Noun and Noun_Description
    noun_noun_description = repeated_noun + row['Noun_Description']
    
    # Concatenate Adjective and Adjective_Synonyms multiple times
    repeated_adjective = (row['Adjective'] + ' ') * adjective_repeats
    adjective_adjective_synonyms = repeated_adjective + row['Adjective_Synonyms']
    
    # Create an InputExample and append it to the list
    train_examples.append(InputExample(texts=[noun_noun_description.strip(), adjective_adjective_synonyms.strip()], label=0.75))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
loss_function = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, loss_function)], epochs=3, warmup_steps=100)

# Save the fine-tuned model
model.save('trained_similarity_model', safe_serialization=False)
