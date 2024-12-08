import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

# Note - boxplot to visualize the distribution: https://builtin.com/data-science/boxplot

# Load the JSON file
with open('./captions-classifier/classification_results.json') as f:
    data = json.load(f)

# Prepare the data
print(f"Total number of entries: {len(data)}")
original_caption_values = []
synthetic_caption_values = []
original_captions = []
synthetic_captions = []

for item in data:
    original_caption_values.append(item["label_probs"][0][0])
    synthetic_caption_values.append(item["label_probs"][0][1])
    original_captions.append(item['original_caption'])
    synthetic_captions.append(item['synthetic_caption'])


data_dict = {
    "original_probability": original_caption_values,
    "synthetic_probability": synthetic_caption_values,
    "original_caption": original_captions,
    "synthetic_caption": synthetic_captions
}

df = pd.DataFrame(data_dict)
print(df.head(8))

# Define the thresholds
thresholds = [0.70, 0.80, 0.90]

# Create a DataFrame to store the counts
counts_list = []

for threshold in thresholds:
    counts_list.append({
        'Threshold': f'Above {int(threshold*100)}%',
        'Caption Type': 'Original Captions\n (SBU Captioned Photo Dataset)',
        'Count': (df['original_probability'] > threshold).sum()
    })
    counts_list.append({
        'Threshold': f'Above {int(threshold*100)}%',
        'Caption Type': 'Synthetic Captions\n (cogvlm2-llama3-chat-19B)',
        'Count': (df['synthetic_probability'] > threshold).sum()
    })

counts_df = pd.DataFrame(counts_list)


# Create bar plots for each threshold
sns.set_style("whitegrid")
g = sns.catplot(
    x='Caption Type', 
    y='Count', 
    hue='Threshold', 
    data=counts_df, 
    kind='bar', 
    palette='pastel',
    height=6, 
    aspect=2
)

g.figure.suptitle("CLIP Model: Image-to-Text Cosine Similarity (scaled x100) Sofmax Probabilities", fontsize=20, fontweight='bold')
g.set_axis_labels("Caption Type", "Count")
g.set_titles(col_template="{col_name}")
g.set_xticklabels(rotation=45)
g.set_ylabels("Count")
plt.subplots_adjust(top=0.85)

# Total count labels to the bars
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

plt.title('Based on a subset of 1,000 samples taken from a dataset containing 10,000 samples', fontsize=16, y=1.02)
plt.show()

g.savefig("./captions-classifier/probability_thresholds.png")