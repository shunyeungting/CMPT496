import csv
import random


# Read in the basic red and green cards into dictionaries
def get_card_sets(filename, colour):
    card_db = {}

    try:
        with open(filename, 'r') as file:
            for line in file:
                primary, secondary  = line.strip().split('&')
                if colour.upper() == "R":
                    related = secondary.strip()
                elif colour.upper() == "G":
                    related = [syn.strip() for syn in secondary.split(',')]
                card_db[primary] = related
        file.close()
    except FileNotFoundError:
        print(f"No file with name {filename}")

    return card_db

# Function to rate a noun-adjective pair
def rate_pair(noun, adjective):
    rating_found = False

    while not rating_found:
        try:
            print(f"Rate how associated '{adjective}' and '{noun}' are (decimals are okay)")
            rating = input("The rating should be between -1 (opposite meaning) and 1 (closely associated), with 0 no association (quit to exit): ")
            if rating == "quit":
                return rating
            rating = float(rating)
        except:
            print("That is not a valid score")
            continue
        if -1 <= rating and rating <= 1:
            rating_found = True
        else:
            print("Please select a number between -1 and 1")

    return rating

def is_humorous(noun, adjective):
    rating_found = False

    while not rating_found:
        try:
            rating = int(input(f"Is the pairing '{adjective} {noun}' funny? (1 for funny, 0 for not funny) "))
        except:
            print("That is not a valid answer")
            continue
        if rating == 1 or rating == 0:
            rating_found = True
        else:
            print("Please select 1 or 0.")
    return rating

def sentiment_rate(noun, adjective):
    rating_found = False

    while not rating_found:
        try:
            rating = int(input(f"Is the pairing '{adjective} {noun}' positive or negative emotionally? (1 for positive, 0 for neutral, -1 for negative) "))
        except:
            print("That is not a valid answer")
            continue
        if rating == 1 or rating == 0 or rating == -1:
            rating_found = True
        else:
            print("Please select 1, 0, or -1.")
    return rating

def is_ironic(noun, adjective):
    rating_found = False
    while not rating_found:
        try:
            rating = int(input(f"Is the pairing '{adjective} {noun}' ironic? (1 for ironic, 0 for not ironic) "))
        except:
            print("That is not a valid answer")
            continue
        if rating == 1 or rating == 0:
            rating_found = True
        else:
            print("Please select 1 or 0.")
    return rating

def is_punny(noun, adjective):
    rating_found = False

    while not rating_found:
        try:
            rating = int(input(f"Is the pairing '{adjective} {noun}' a pun? (1 for pun, 0 for no pun) "))
        except:
            print("That is not a valid answer")
            continue
        if rating == 1 or rating == 0:
            rating_found = True
        else:
            print("Please select 1 or 0.")
    return rating

def main():
    nouns_with_descriptions = get_card_sets("RED.txt", "R")
    adjectives_with_synonyms = get_card_sets("GREEN.txt", "G")

    # Number of samples you want to generate
    num_samples = 1000

    # Checking if the file already exists
    file_exists = False
    try:
        with open('new_dataset.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            if len(list(reader)) > 0:
                file_exists = True
    except FileNotFoundError:
        pass

    # Generating and storing the dataset
    with open('new_dataset.csv', 'a', newline='') as csvfile:
        fieldnames = ['Noun', 'Noun_Description', 'Adjective', 'Adjective_Synonyms', 'Rating', 'Is_Humorous', 'Sentiment', 'Is_Ironic', 'Is_Punny']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for i in range(num_samples):
            noun = random.choice(list(nouns_with_descriptions.keys()))
            adjective = random.choice(list(adjectives_with_synonyms.keys()))
            rating = rate_pair(noun, adjective)
            if rating == 'quit':
                break
            is_funny = is_humorous(noun, adjective)
            sentiment = sentiment_rate(noun, adjective)
            ironic = is_ironic(noun, adjective)
            punny = is_punny(noun, adjective)

            # Write the data to the CSV file
            writer.writerow({
                'Noun': noun,
                'Noun_Description': nouns_with_descriptions[noun],
                'Adjective': adjective,
                'Adjective_Synonyms': ', '.join(adjectives_with_synonyms[adjective]),
                'Rating': rating,
                'Is_Humorous': is_funny,
                'Sentiment': sentiment,
                'Is_Ironic': ironic,
                'Is_Punny': punny
            })

    print("Dataset updated successfully.")

if __name__ == "__main__":
    main()