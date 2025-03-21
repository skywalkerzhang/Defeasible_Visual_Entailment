# Defeasible Visual Entailment

This is the official code implementation for the AAAI 2025 paper *"Defeasible Visual Entailment: Benchmark, Evaluator, and Reward-Driven Optimization"*.

## Dataset

- **Image Data**: The image dataset can be downloaded by filling out the form at [this link](https://forms.illinois.edu/sec/229675).
- **Text Data**: The text data has already been uploaded. You can access it from the repository.

We would like to thank the creators of the following datasets for their contributions:

- **Flickr30k**: A large-scale image dataset with natural language descriptions.
- **SNLI (Stanford Natural Language Inference)**: A dataset for developing and evaluating models for natural language inference.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/Defeasible_Visual_Entailment.git
    cd Defeasible_Visual_Entailment
    ```

2. **Install the necessary dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Dataset Details

The dataset consists of images paired with text captions. These pairs are annotated for visual entailment tasks, where the model determines whether the image **entails, contradicts, or is neutral** to the given text.

| Split           | Number of Samples | Weakener Count | Strengthener Count | Unique Images |
|----------------|------------------|---------------|----------------|---------------|
| **Training**   | 93,082            | 46,541        | 46,541         | 9,507         |
| **Validation** | 1,888             | 944           | 944            | 195           |
| **Test**       | 1,972             | 986           | 986            | 203           |

Each sample contains:
- An **image premise**
- A **text hypothesis**
- A textual **update** that either strengthens or weakens the hypothesis
## Usage

To train the model, run the following command:

```bash
python visual_text_training.py \
  --train_csv_file ../Data/DVE_train.csv \
  --val_csv_file ../Data/DVE_dev.csv \
  --image_dir ../Data/flickr30k_images \
  --epochs 20 \
  --lr 5e-6 \
  --batch_size 32 \
  --wandb_project "Defeasible_Visual_Entailment" \
  --output_model "DVE_model.pth" \
  --gpu 0 \
  --classification_weight 0.9 \
  --use_classification_head
```

## Model

The model is designed to handle visual and textual inputs, combining them to predict the relationship (entailment, contradiction, or neutral) between the image and the caption.

The model integrates a reasoning evaluator, which assesses the strength of updates and their impact on visual entailment tasks. This allows for reward-driven optimization, improving model performance over time.

### Model Link

The pre-trained evaluator model can be downloaded from Hugging Face:

➡ **[DVE Evaluator Model](https://huggingface.co/skywalkerzhang19/DVE_evaluator/resolve/main/evaluator_weights.pth?download=true)**

You can also download it via `wget` or `curl`:

```bash
wget "https://huggingface.co/skywalkerzhang19/DVE_evaluator/resolve/main/evaluator_weights.pth?download=true" -O evaluator_weights.pth
```
or

```bash
curl -L "https://huggingface.co/skywalkerzhang19/DVE_evaluator/resolve/main/evaluator_weights.pth?download=true" -o evaluator_weights.pth
```

## Evaluation
To evaluate the model, use the following command:
```bash
python visual_text_inference.py \
  --test_csv_file ../Data/DVE_test.csv \
  --image_dir ../Data/flickr30k_images \
  --model_path evaluator_weights.pth \
  --output_file "test_results.csv" \
  --gpu 0 \
  --test_batch_size 64
```

For running inference on specific data using inference_demo.py, execute the following command:
```bash
python inference_demo.py \
  --image_path path/to/your/image.jpg \
  --text "Your hypothesis text" \
  --update "Your update text" \
  --model_path Evaluator/evaluator_weights.pth \
  --output_file "inference_results.txt" \
  --gpu 0
```
