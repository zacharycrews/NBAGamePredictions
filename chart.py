import numpy as np
import matplotlib.pyplot as plt

labels = ('Random Baseline', 'Majority', 'Sentiment', 'Naive Bayes', 'Log. Reg.', 'SVM', 'FForward-NN')
f1_score = [50,53,71,74,69,76,62]
accuracy = [50,53,55,62,60,61,57]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, f1_score, width, label='f1-score')
rects2 = ax.bar(x + width/2, accuracy, width, label='accuracy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Accuracy & F1_score')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=4)
ax.bar_label(rects2, padding=4)

fig.tight_layout()

plt.show()
