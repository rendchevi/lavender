### ViPE: Visualize Pretty-much Everything
The training script in this folder is from the cool work by [Hassan Shahmohammadi et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.333/).

Go check-out their [paper](https://aclanthology.org/2023.emnlp-main.333/) and [repository](https://github.com/Hazel1994/ViPE) for further details.


### Training
To train the ViPE model, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone git@github.com:Hazel1994/ViPE.git
   ```
2. **Setup**
   Ensure Python and required dependencies are installed.
   ```bash
   virtualenv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   cd training
   ```

3. **Run Training**
   Use the following command to start training:
   ```bash
   python train.py --model_name gpt2-medium --data_set_dir path/to/lyric_canvas.csv --check_path /save/vipe/here/ --batch_size 32 --epochs 5 --context_length 7 
   ```
   
   - `model_name`: Choose between 'gpt2-medium' or 'gpt2'
   - `data_set_dir`: Path to the lyricCanvas dataset
   - `check_path`: Path to save the trained model
   - `batch_size`, `epochs`, `learning_rate`, `warmup_steps`, `context_length`: Training hyperparameters
   - `device`: 'cuda' or 'cpu'

We have used 9 A100 GPUs for training gpt2-medium with a context size of 7. Modify the code as required
