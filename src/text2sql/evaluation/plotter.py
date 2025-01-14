import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(gen_models, accuracies, labels, plot_file_name):
    # Ensure all inputs are of the same length
    assert len(accuracies) == len(labels), "Number of accuracy lists must match number of labels"

    # Set width of bar and group spacing
    bar_width = 0.8 / len(accuracies)  # Adjust bar width based on number of metrics
    group_spacing = 0.2

    # Calculate the width of each group
    group_width = bar_width * len(accuracies) + group_spacing

    # Set position of bars on X axis
    indices = np.arange(len(gen_models))
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Dynamic color palette
    colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(accuracies)))

    # Plot bars for each accuracy metric
    for i, (acc_list, label) in enumerate(zip(accuracies, labels)):
        r = indices * group_width + i * bar_width
        bars = plt.bar(r, acc_list, color=colors[i], width=bar_width, label=label)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.1f}', ha='center', va='bottom')

    # Add labels and title
    plt.xlabel('Model Names')
    plt.ylabel('Accuracy (%)')
    # plt.title('Model Accuracy Comparison')
    plt.xticks(indices * group_width + (len(accuracies) - 1) * bar_width / 2, gen_models)

    # Set y-axis to start at 0 and end at 100
    plt.ylim(0, 100)

    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add legend
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')

    # Close the plot to free up memory
    plt.close()

    print(f"Plot has been saved as '{plot_file_name}'")
