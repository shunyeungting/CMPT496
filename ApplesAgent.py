#!/usr/bin/env python3

import torch
import torch.nn as nn
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import sys

class LiteralJudge:
    def __init__(self):
        self.model_name = "./Models/Association"
        self.model = SentenceTransformer(self.model_name)

    def get_embedding(self, text):
        embed = self.model.encode(text, convert_to_tensor=True)
        return embed

    def cosine_sim(self, embed1, embed2):
        return util.pytorch_cos_sim(embed1, embed2)

    # Returns single semantic similarity rating for given red and green card embeddings
    def single_sem_sim_rating(self, embed_red, embed_green):
        similarity_matrix = self.cosine_sim(embed_red, embed_green)
        return similarity_matrix.item()  # Convert tensor to scalar

    # Gets Semantic Similarity ratings for bulk comparisons, sorted from highest to lowest
    def bulk_sem_sim_ratings(self, reds, greens):
        ratings = []
        for red in reds:
            embed_red = self.get_embedding(red)
            for green in greens:
                embed_green = self.get_embedding(green)
                rating = self.single_sem_sim_rating(embed_red, embed_green)
                ratings.append((rating, red, green))

        ratings.sort(reverse=True, key=lambda x: x[0])
        return ratings

    # Finds the similarity between all reds in hand and the green card in play
    def eval_hand(self, reds, green):
        best_score = float('-inf')
        best_card = None
        worst_score = float('inf')
        worst_card = None
        embed_green = self.get_embedding(green)

        for red in reds:
            embed_red = self.get_embedding(red)
            rating = self.single_sem_sim_rating(embed_red, embed_green)
            if rating > best_score:
                best_score = rating
                best_card = red
            if rating < worst_score:
                worst_score = rating
                worst_card = red
        return [best_card, worst_card]


class HumorJudge:
    def __init__(self):
        self.model_name = "./Models/PretrainedHumor"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  

        # Load the locally saved trained model
        self.model_path = "./Models/TrainedHumor.pth"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.classifier = nn.Linear(self.model.config.hidden_size, 1)  # Replace the final classifier with a single neuron
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

    # Encodes (tokenizes) given primary text (Main Noun/Adjective) and secondary text (following related words/sentence)
    def encode(self, red_card, green_card):
        encoded = self.tokenizer(red_card, green_card, return_tensors="pt")
        return encoded

    # Finds the confidence score that a given red & green card combination is humorous
    def humor_confidence(self, red_card, green_card):
        enc_text = self.encode(red_card, green_card)
        with torch.no_grad():                                           
            output = self.model(**enc_text)
            logits = output.logits.squeeze()  # Squeeze the logits tensor
            humorous_confidence = torch.sigmoid(logits).item()  # Apply sigmoid to get confidence score
        return humorous_confidence

    # Gets the confidence score of each red card for the green card in play
    def eval_hand(self, reds, green):
        best_score = float('-inf')
        best_card = None

        for red in reds:
            rating = self.humor_confidence(red, green)
            if rating > best_score:
                best_score = rating
                best_card = red

        return best_card


class ApplesToApples:
    def __init__(self, players, max_points, green_file, red_file):
        # Initialize players, scores, and cards.
        self.literal = LiteralJudge()
        self.humor = HumorJudge()
        self.contrarian = None
        
        self.judge_types = [self.literal, self.humor, self.contrarian] # The different types of judges our agent will track

        self.players = [player+1 for player in range(int(players))]
        self.current_judge = 0  # Index of the current judge
        self.judge_index = None # Tracks when we are the judge
        self.player_judges = [player for player in range(int(players))] 
        self.player_judge_types = [[0,0,0] for player in range(int(players))]

        self.max_points = int(max_points)
        self.scores = {player: 0 for player in self.players}  # Initialize scores for each player
        self.hand = [] # holds the cards currently in agent hand
        self.role = None # either 1 - player, or anything else - judge

        self.load_sets(red_file, green_file)

    # Load both the basic set and newly provided extended set
    def load_sets(self, red_file, green_file):
        self.green_cards = self.get_card_sets("./WordSets/GREEN.txt", "G")
        self.red_cards = self.get_card_sets("./WordSets/RED.txt", "R")

        self.green_cards.update(self.get_card_sets(green_file, "G"))
        self.red_cards.update(self.get_card_sets(red_file, "R"))

    def game_loop(self):
        print("I assume I am player 1!")
        print("The other players are:", end=" ")
        print(*self.players[1:], sep = ', ')
        print("Please enter the red cards dealt to me, one by one.")
        for i in range(1,8):
            self.hand.append(input(f"Card {i}: ")) # Gets a hand of 7 cards

        # Continuously play rounds until a player has reached the point limit
        while self.max_points not in self.scores.values():
            self.new_round()
            self.update_judge_index()
            
    # Called at the start of each new round
    def new_round(self):
        print("---------------NEW ROUND------------------")
        self.role = input("What is my role in this round? (1-player; anything else - judge): ")
        if self.role == "1":
            self.is_player()
        else:
            if not self.judge_index:
                self.judge_index = self.current_judge
            self.is_judge()

    # Plays through one hand as the player
    def is_player(self):
        player_cards = []
        green_card = input("Please enter the green card selected by the judge: ")
        if '&' in green_card:
            green_card = green_card.split('&')[0]
        self.green_cards.pop(green_card)


        judge_type_i = self.current_judge_personality() # Finds index of the most likely current judge type based on previous rounds
        judge_type = self.judge_types[judge_type_i] # Judge type at that index

        if judge_type_i == 0:
            played_card = judge_type.eval_hand(self.hand, green_card)[0] # Judge is a literal type, finds direct association
        elif judge_type_i == 1:
            played_card = judge_type.eval_hand(self.hand, green_card) # Judge is a humor type, finds funniest pair
        elif judge_type_i == 2:
            played_card = judge_type.eval_hand(self.hand, green_card)[1] # Judge is a contrarian, finds card with least relation
        
        player_cards.append(played_card)
        self.hand.remove(played_card)

        print("The red card I play is:")
        print(played_card)
        winner = int(input("Please tell me which player won (number): "))
        self.scores[winner] += 1
        winning_card = played_card

        print("Now tell me the red cards the other players selected")
        for i in range(2, len(self.players)+1):
            player_card = input(f"Player {i}'s Card (if judge, 'None'): ")
            if player_card != 'None':
                if i == winner:
                    winning_card = player_card
                player_cards.append(player_card)

        round_judge = self.detect_judge_type(player_cards, winning_card, green_card)
        self.update_judges(round_judge)

        dealt_card = input("Now tell me the new red card I was dealt ")
        self.hand.append(dealt_card)

        return

    # Plays through one hand as the judge
    def is_judge(self):
        player_cards = []
        green_card = random.choice(list(self.green_cards.items()))
        self.green_cards.pop(green_card[0])
        
        print("The green card I selected is:")

        print(green_card[0])
        print("Now tell me the red cards the other players selected")
        for i in range(len(self.players)-1):
            player_cards.append(input(f"Card {i+1}: "))
        
        winning_card = self.literal.eval_hand(player_cards, green_card[0])[0]

        print("The winning card is: ")
        print(winning_card)
        winner = int(input("Please tell me which player won (number): "))
        self.scores[winner] += 1

        return

    def current_judge_personality(self):
        current = self.current_judge
        return self.player_judge_types[current].index(max(self.player_judge_types[current]))

    # Detects the personality of the current judge
    def detect_judge_type(self, player_cards, winning_card, green_card):
        literal_contrarian = self.literal.eval_hand(player_cards, green_card)
        best_literal = literal_contrarian[0]
        best_humor = self.humor.eval_hand(player_cards, green_card)
        best_contrarian = literal_contrarian[1]

        if winning_card == best_literal:
            judge = 0
        elif winning_card == best_humor:
            judge = 1
        elif winning_card == best_contrarian:
            judge = 2
        else:
            judge = None

        return judge
    
    def update_judges(self, judge_type):
        if judge_type == None: # The card selected did not match a judge archetype, therefore do not change our judge information
            return
        else:
            self.player_judge_types[self.current_judge][judge_type] += 1 # Update the count for the judge archetype that was used last round
        return

    def play_game(self):
        self.game_loop()
        return
        
    def update_judge_index(self):
        # Rotate the judge role among players
        self.current_judge = (self.current_judge + 1) % len(self.players)
        return
    
    def get_card_sets(self, filename, colour):
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


def main():    
    players = sys.argv[1]
    points = sys.argv[2]
    red_file = sys.argv[3]
    green_file = sys.argv[4]
    
    # players = 3
    # points =3 
    # red_file = "REDtest.txt"
    # green_file = "GREENtest.txt"

    game = ApplesToApples(players, points, red_file, green_file)

    game.play_game()


if __name__ == "__main__":
    main()