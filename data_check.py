from datasets import load_dataset

# load dataset
dataset = load_dataset('metaeval/offensive-humor', split='train')

# print the first row
print(dataset[0])
