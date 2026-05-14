import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

try:
	eval_df = pd.read_csv("results_eval.csv")
	print(f"Loaded {len(eval_df)} evaluated questions")
except FileNotFoundError:
	print("File not founded. Run evals.py")
	sys.exit()

questions = range(1, len(eval_df) + 1)
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Scores', fontsize=18, fontweight='bold')

metrics = [
	('faithfulness', 'Faithfulness', axs[0, 0], 'navy'),
	('answer_relevancy', 'Answer Relevancy', axs[0, 1], 'darkgreen'),
	('context_precision', 'Context Precision', axs[1, 0], 'indigo'),
	('context_recall', 'Context Recall', axs[1, 1], 'maroon')
]

for col, title, ax, color in metrics:
	mean = eval_df[col].mean()
	ax.plot(questions, eval_df[col], marker='o', linestyle='-', color=color, alpha=0.7, label='Single question')
	ax.axhline(mean, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
	ax.set_title(title, fontsize=14, pad=10)
	ax.set_ylim(-0.05, 1.1)
	ax.set_xlabel("Question number", fontsize=10)
	ax.set_ylabel('Score', fontsize=10)
	ax.grid(True, linestyle='--', alpha=0.5)
	ax.legend(loc='lower right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
file_name = 'rag_scores.png'
plt.savefig(file_name, bbox_inches='tight', dpi=300)
print(f"Plot saved as '{file_name}")