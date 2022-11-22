from ReadCSV import games, labels
import random

random_baselines = []
home_team_baselines = []

for i in range(10000):
    
    # Random baseline
    num_correct = 0
    choice = 0
    for winner in labels:
        choice = random.randint(0,1)
        if choice == winner:
            num_correct += 1
    random_baselines.append(num_correct/len(labels))

    # Home team baseline
    num_correct = 0
    for winner in labels:
        if winner == 1:
            num_correct += 1
    home_team_baselines.append(num_correct/len(labels))

print("Random baseline:", sum(random_baselines)/len(random_baselines))
print("Home team baseline:", sum(home_team_baselines)/len(home_team_baselines))
