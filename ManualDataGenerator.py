import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import csv
import random

# Initialize global variables
nouns_with_descriptions = {}
adjectives_with_synonyms = {}
output_csv_filename = ""
current_sample = 0
num_samples = 0
writer = None
csvfile = None  # New global variable to keep the file open

# Read red and green card words
def get_card_sets(filename, color):
    card_db = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                primary, secondary = line.strip().split('&')
                if color.upper() == "R":
                    related = secondary.strip()
                elif color.upper() == "G":
                    related = [syn.strip() for syn in secondary.split(',')]
                card_db[primary] = related
        file.close()
    except FileNotFoundError:
        print(f"File not found: {filename}")
    return card_db

# Load word files
def load_file(color):
    filename = filedialog.askopenfilename(title=f"Select {color} card file", filetypes=[("Text files", "*.txt")])
    if filename:
        if color == 'Red':
            nouns_with_descriptions.update(get_card_sets(filename, 'R'))
            label_red_file.config(text=f"Red card file: {filename.split('/')[-1]} loaded")
        elif color == 'Green':
            adjectives_with_synonyms.update(get_card_sets(filename, 'G'))
            label_green_file.config(text=f"Green card file: {filename.split('/')[-1]} loaded")
        messagebox.showinfo("File Loaded", f"{color} card file loaded successfully!")

# Set output CSV file name
def set_output_csv_name():
    global output_csv_filename
    output_csv_filename = filedialog.asksaveasfilename(title="Save as", filetypes=[("CSV files", "*.csv")], defaultextension=".csv")
    if output_csv_filename:
        label_csv_file.config(text=f"CSV file: {output_csv_filename.split('/')[-1]}")
        messagebox.showinfo("File Name Set", "Output CSV file name set successfully!")

# Submit the current evaluation and update to the next word pair
def submit_and_update():
    global current_sample

    results = {
        "rating": rating_var.get(),
        "humor": humor_var.get(),
        "sentiment": sentiment_var.get(),
        "irony": irony_var.get(),
        "punny": punny_var.get()
    }
    
    writer.writerow({
        'Noun': current_noun,
        'Noun_Description': nouns_with_descriptions[current_noun],
        'Adjective': current_adjective,
        'Adjective_Synonyms': ', '.join(adjectives_with_synonyms[current_adjective]),
        'Rating': results['rating'],
        'Is_Humorous': results['humor'],
        'Sentiment': results['sentiment'],
        'Is_Ironic': results['irony'],
        'Is_Punny': results['punny']
    })
    
    current_sample += 1
    if current_sample < num_samples:
        update_evaluation_window()
    else:
        messagebox.showinfo("Completed", "Dataset generation completed!")
        eval_window.destroy()
        close_csv_file()  # Close the file after all samples are processed

# Update the evaluation window content
def update_evaluation_window():
    global current_noun, current_adjective
    
    current_noun = random.choice(list(nouns_with_descriptions.keys()))
    current_adjective = random.choice(list(adjectives_with_synonyms.keys()))
    
    eval_window.title(f"'{current_adjective}' vs '{current_noun}'")
    
    rating_var.set(0)
    humor_var.set(0)
    sentiment_var.set(0)
    irony_var.set(0)
    punny_var.set(0)

# Generate dataset
def generate_dataset():
    global current_sample, num_samples, writer, csvfile, eval_window
    
    num_samples = int(entry_samples.get())  # Get the number of samples from user input
    
    csvfile = open(output_csv_filename, 'w', newline='')  # Open the file
    fieldnames = ['Noun', 'Noun_Description', 'Adjective', 'Adjective_Synonyms', 'Rating', 'Is_Humorous', 'Sentiment', 'Is_Ironic', 'Is_Punny']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    current_sample = 0

    # Create the evaluation window
    eval_window = tk.Toplevel(root)
    eval_window.title("Evaluation Window")
    eval_window.geometry("800x400")
    
    tk.Label(eval_window, text=f"Association Rating").pack(pady=5)
    global rating_var
    rating_var = tk.DoubleVar(value=0)
    tk.Scale(eval_window, from_=-1, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=rating_var).pack(pady=5)

    global humor_var
    humor_var = tk.IntVar(value=0)
    tk.Checkbutton(eval_window, text=f"Is it funny?", variable=humor_var).pack(pady=5)

    global sentiment_var
    sentiment_var = tk.IntVar(value=0)
    sentiment_frame = tk.Frame(eval_window)
    sentiment_frame.pack(pady=5)
    tk.Label(sentiment_frame, text=f"Sentiment").pack(side=tk.LEFT)
    tk.Radiobutton(sentiment_frame, text="Positive", variable=sentiment_var, value=1).pack(side=tk.LEFT)
    tk.Radiobutton(sentiment_frame, text="Neutral", variable=sentiment_var, value=0).pack(side=tk.LEFT)
    tk.Radiobutton(sentiment_frame, text="Negative", variable=sentiment_var, value=-1).pack(side=tk.LEFT)

    global irony_var
    irony_var = tk.IntVar(value=0)
    tk.Checkbutton(eval_window, text=f"Ironic?", variable=irony_var).pack(pady=5)

    global punny_var
    punny_var = tk.IntVar(value=0)
    tk.Checkbutton(eval_window, text=f"Punny?", variable=punny_var).pack(pady=5)

    tk.Button(eval_window, text="Submit", command=submit_and_update).pack(pady=10)

    # Initialize the window content
    update_evaluation_window()

# Close the file
def close_csv_file():
    global csvfile
    if csvfile:
        csvfile.close()
        csvfile = None

# Initialize the main window
root = tk.Tk()
root.title("Card Rating Setup")
root.geometry("400x400")

# Create and place widgets
label_red_file = tk.Label(root, text="Red card file: Not loaded")
label_red_file.pack(pady=5)
button_load_red = tk.Button(root, text="Load Red Card File", command=lambda: load_file('Red'))
button_load_red.pack(pady=5)

label_green_file = tk.Label(root, text="Green card file: Not loaded")
label_green_file.pack(pady=5)
button_load_green = tk.Button(root, text="Load Green Card File", command=lambda: load_file('Green'))
button_load_green.pack(pady=5)

label_csv_file = tk.Label(root, text="CSV file: Not set")
label_csv_file.pack(pady=5)
button_set_csv = tk.Button(root, text="Set Output CSV Filename", command=set_output_csv_name)
button_set_csv.pack(pady=5)

label_samples = tk.Label(root, text="Number of Samples:")
label_samples.pack(pady=5)
entry_samples = tk.Entry(root)
entry_samples.insert(0, "20")
entry_samples.pack(pady=5)

button_generate = tk.Button(root, text="Generate Dataset", command=generate_dataset)
button_generate.pack(pady=20)

root.mainloop()
