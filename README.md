# Player Segregation

This project segregates badminton player images into individual player classes using computer vision techniques.

Submission by Om Vibhandik (IIT2021094)

## Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- Pillow
- NumPy
- tqdm

You can install the required packages using:

```
pip install torch torchvision scikit-learn Pillow numpy tqdm
```

## Usage

1. Ensure your input images are placed in the `two_players_bot` and `two_players_top` folders.

2. Run the script:
   ```
   ./execute.sh
   ```
3. The script will create an `output` folder containing four subfolders, each representing a player class.

## How It Works

1. The script uses a pre-trained ResNet18 model as a feature extractor.
2. It processes images from both input folders, extracting features for each image.
3. K-means clustering is applied to separate the players in each folder.
4. Images are copied to the appropriate player folders in the output directory.

## Performance

- The script utilizes GPU acceleration if available, falling back to CPU if not.
- Progress bars are displayed to show the status of feature extraction and image segregation.

## Troubleshooting

If you encounter any issues:
- Ensure all required packages are installed correctly.
- Check that your input folders contain valid image files (PNG, JPG, or JPEG).
- Verify that you have sufficient disk space for the output folder.
