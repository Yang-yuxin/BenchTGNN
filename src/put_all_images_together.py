import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# The number of images
n_images = 8

# Create a figure and an array of axes to plot each image
fig, axs = plt.subplots(n_images, 1, figsize=(10, 20))


datasets = ['CollegeMsg', 'REDDIT', 'uci']
settings = ['GraphMixer_re_embed',
            'GraphMixer_re_gru',
            'GraphMixer_uni_embed',
            'GraphMixer_uni_gru',
            'TGAT_re_embed',
            'TGAT_re_gru',
            'TGAT_uni_embed',
            'TGAT_uni_gru',]
# Loop through each small figure and add it to the subplot
for dataset in datasets:
    
    for i in range(n_images):
        # Load the image
        img = Image.open(f'{dataset}_node_degree_plot_pos_{settings[i]}.png')
        
        # Convert the image to an array and display it
        axs[i].imshow(np.asarray(img))
        
        # Remove axis for cleaner look
        axs[i].axis('off')

    # Adjust spacing between the images
    plt.subplots_adjust(hspace=-0.2)

    plt.suptitle(f'{dataset}')
    # Adjust layout
    plt.tight_layout()

    # Save the combined figure
    plt.savefig(f'{dataset}_combined_figure.png', dpi=200)


