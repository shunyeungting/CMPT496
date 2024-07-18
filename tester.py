import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import csv
import random
from transformers import pipeline

# Global variables
card_files = {'RED': '', 'GREEN': ''}
models = {'Humorous': None, 'Sentiment': None, 'Ironic': None, 'Punny': None}
output_csv_filename = ""
nouns_with_descriptions = {}
adjectives_with_synonyms = {}
pipes = {}

# Initialize main window
root = tk.Tk()
root.title("Card Rating Setup")
root.geometry("400x350")

def load_file(color, button):
    filename = filedialog.askopenfilename(title=f"Select {color} card file", filetypes=[("Text files", "*.txt")])
    if filename:
        card_files[color] = filename
        if color == 'RED':
            nouns_with_descriptions.update(load_card_sets(filename, color))
        elif color == 'GREEN':
            adjectives_with_synonyms.update(load_card_sets(filename, color))
        button.config(text=f"{color} Card: {filename.split('/')[-1]} loaded")
        messagebox.showinfo("File Loaded", f"{color} card file loaded successfully!")

def load_card_sets(filename, colour):
    card_db = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                primary, secondary = line.strip().split('&')
                if colour == "RED":
                    card_db[primary] = secondary.strip()
                elif colour == "GREEN":
                    card_db[primary] = [syn.strip() for syn in secondary.split(',')]
    except FileNotFoundError:
        print(f"No file with name {filename}")
    return card_db

def set_model_name(feature, button):
    model_name = simpledialog.askstring("Model Name", f"Enter the model name for '{feature}':")
    if model_name:
        try:
            pipes[feature] = pipeline("text-classification", model=model_name)
            button.config(text=f"{feature} Model: {model_name}")
            messagebox.showinfo("Model Loaded", f"Pipeline for '{feature}' loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Loading Error", str(e))

def set_output_csv_name(button):
    global output_csv_filename
    output_csv_filename = filedialog.asksaveasfilename(title="Save as", filetypes=[("CSV files", "*.csv")], defaultextension=".csv")
    if output_csv_filename:
        button.config(text=f"CSV File: {output_csv_filename.split('/')[-1]}")
        messagebox.showinfo("File Name Set", "Output CSV file name set successfully!")

# GUI Elements Setup
button_load_red = tk.Button(root, text='Load Red Card File', command=lambda: load_file('RED', button_load_red))
button_load_red.pack(pady=5)

button_load_green = tk.Button(root, text='Load Green Card File', command=lambda: load_file('GREEN', button_load_green))
button_load_green.pack(pady=5)

button_model_humorous = tk.Button(root, text="Set Model for 'Is Humorous'", command=lambda: set_model_name('Is_Humorous', button_model_humorous))
button_model_humorous.pack(pady=5)

button_model_sentiment = tk.Button(root, text="Set Model for 'Sentiment'", command=lambda: set_model_name('Sentiment', button_model_sentiment))
button_model_sentiment.pack(pady=5)

button_model_ironic = tk.Button(root, text="Set Model for 'Is Ironic'", command=lambda: set_model_name('Is_Ironic', button_model_ironic))
button_model_ironic.pack(pady=5)

button_model_punny = tk.Button(root, text="Set Model for 'Is Punny'", command=lambda: set_model_name('Is_Punny', button_model_punny))
button_model_punny.pack(pady=5)

button_set_csv = tk.Button(root, text='Set Output CSV Filename', command=lambda: set_output_csv_name(button_set_csv))
button_set_csv.pack(pady=5)

button_proceed = tk.Button(root, text='Proceed', command=lambda: proceed())
button_proceed.pack(pady=10)

def proceed():
    if not all(card_files.values()) or not output_csv_filename or not all(pipes.values()):
        messagebox.showerror("Error", "Please load all files and models and set the output CSV filename.")
        return
    generate_dataset()
    messagebox.showinfo("Success", "Dataset generated successfully.")

def generate_dataset():
    with open(output_csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Noun', 'Noun_Description', 'Adjective', 'Adjective_Synonyms', 'Is_Humorous', 'Sentiment', 'Is_Ironic', 'Is_Punny']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(100):  # Generate 100 samples
            noun = random.choice(list(nouns_with_descriptions.keys()))
            adjective = random.choice(list(adjectives_with_synonyms.keys()))
            text = f"{adjective} {noun}"
            humorous = predict_using_model('Is_Humorous', text)
            sentiment = predict_using_model('Sentiment', text)
            ironic = predict_using_model('Is_Ironic', text)
            punny = predict_using_model('Is_Punny', text)
            writer.writerow({
                'Noun': noun,
                'Noun_Description': nouns_with_descriptions[noun],
                'Adjective': adjective,
                'Adjective_Synonyms': ', '.join(adjectives_with_synonyms[adjective]),
                'Is_Humorous': humorous,
                'Sentiment': sentiment,
                'Is_Ironic': ironic,
                'Is_Punny': punny
            })

def predict_using_model(feature, text):
    if feature in pipes:
        try:
            result = pipes[feature](text)
            return result[0]['label']
        except Exception as e:
            print(f"Error using model {feature}: {str(e)}")
    return 'N/A'  # Default prediction if model fails or is not loaded

root.mainloop()