import ApplesAgentClean2
import random


def load_sets(red_file, green_file):
    green_cards = get_card_sets("./WordSets/GREEN.txt", "G")
    red_cards = get_card_sets("./WordSets/RED.txt", "R")

    green_cards.update(get_card_sets(green_file, "G"))
    red_cards.update(get_card_sets(red_file, "R"))

    return green_cards, red_cards

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
    
def main():
    red_file = "REDExt.txt"
    green_file = "GREENExt.txt"
    max_points = 10
    num_players = 2
    players = []
    discarded = {}

    # Initialize game environment
    sets = load_sets(red_file, green_file)
    green_cards, red_cards = sets[0], sets[1]

    # Populates a list with num_players instances of the agent
    # for n in range(num_players):
    #     players[n] = Agent(num_players)
    
    # Initialize hands for the players
    while True:

        # Round setup
        green_card = random.choice(list(green_cards.items()))
        discarded[green_card[0]] = green_card[1]
        print(discarded)

        break

        # player1_card = player1.play_card(green_card)
        # player2_card = player2.play_card(green_card)

        # # Judge's decision
        # judge = Agent(2, [...])  # Create judge instance
        # judge_card = judge.play_card(green_card)

        # # Scoring
        # # Determine the winner and update scores

if __name__ == "__main__":
    main()