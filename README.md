# SmolLM2 to DeepSeek Architecture Conversion

## Overview
This project involves converting the SmolLM2 architecture into the DeepSeek architecture, which includes the implementation of Multi-Head Latent Attention (MHLA) and Mixture of Experts (MoE) with loss-less load balancing. The model is trained on a text dataset and generates text outputs based on the learned patterns.

## Features
- **Multi-Head Latent Attention (MHLA)**: 
  - The model implements latent transformations for each attention head through the `latent_query` and `latent_key` layers. 
  - This allows the model to capture richer interactions and dependencies in the input data compared to standard multi-head attention.

- **Mixture of Experts (MoE)**: 
  - The architecture includes multiple FeedForward sub-layers, each potentially specializing in different aspects of the data.
  - The routing mechanism uses `gates = F.softmax(routing_logits, dim=-1)` to decide which expert to use for each token, enabling dynamic selection based on the input.

- **Loss-less Load Balancing**: 
  - The `update_bias_terms` method adjusts `self.routing_bias` to ensure that each expert receives roughly equal usage during training.
  - This approach eliminates the need for an additional explicit loss function, optimizing expert utilization without compromising performance.

## Project Structure
- `train.py`: The main training script that sets up the model, data, and training loop.
- `model.py`: Contains the model architecture, including the DeepSeekAttention and Mixture of Experts layers.
- `loadconfig.py`: Loads the configuration from a YAML file.
- `streamingdata.py`: Handles the streaming of text data for training.
- `logs.txt`: Training logs showing the loss and generated outputs.

## Model Configuration
- hidden_size: 576
- num_hidden_layers: 30
- num_attention_heads: 9
- intermediate_size: 1536
- vocab_size: 49152
- max_position_embeddings: 2048

## Installation
To run this project, ensure you have the required packages installed. You can install them using the following command:

```bash
!pip install torch torchvision transformers datasets
```

## Training
To train the model, run the `train.py` script. The model will be trained for a total of 12001 steps, and it will generate text samples periodically.

```bash
python train.py
```

## Training Logs
Training for 12,000 steps (Step 0 to Step 12000)
The training logs can be found in [Training Logs](logs.txt). Below are some key entries from the logs:

Step 11999/12001 | Loss: 3.1403
Step 12000/12001 | Loss: 3.7546

 Generating sample at step 12000 
Prompt: Solar system is a
Generated: Solar system is a system system system system system system system system system system system system system system system system system system system system system system system system system system system system system system

Step 12001/12001 | Loss: 3.7798

Training complete!
Best loss achieved: 2.9509

## Sample Outputs
During training, 5 sample outputs were generated every 2000 steps. The final output at step 12000 is included in the logs.

## Best Loss
The best loss is **2.9509**.

## Conclusion
This project demonstrates the conversion of the SmolLM2 architecture into the DeepSeek architecture, showcasing the implementation of advanced attention mechanisms and expert routing. The model's performance can be further improved with additional training and hyperparameter tuning.




