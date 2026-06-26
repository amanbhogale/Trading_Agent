import nbformat as nbf
import os

# Files to combine in order
notebooks = [
    '02_data_pipeline.ipynb',
    '03_lstm_model.ipynb',
    '04_dqn_agent_rainbow.ipynb',
    '05_backtest_engine.ipynb',
    '06_results_dashboard.ipynb'
]

base_path = '/home/zombie/Documents/Trading_Agent/deep_learning_backtest/'

master_nb = nbf.v4.new_notebook()

# Add Colab cell first
colab_cell = """# Connect to Google Drive and Set up Environment
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
BASE_DIR = '/content/drive/MyDrive/Trading_Agent'
os.makedirs(BASE_DIR, exist_ok=True)

# Create standard subdirectories
for sub in ['configs', 'models', 'data', 'results']:
    os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

# Change directory
os.chdir(BASE_DIR)
print("✅ Connected to Drive and set working directory to:", os.getcwd())
"""
master_nb.cells.append(nbf.v4.new_code_cell(colab_cell))

# Iterate and append
for nb_file in notebooks:
    full_path = os.path.join(base_path, nb_file)
    if os.path.exists(full_path):
        master_nb.cells.append(nbf.v4.new_markdown_cell(f"# --- Section: {nb_file} ---"))
        with open(full_path, 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
            # Filter out empty cells
            for cell in nb.cells:
                if cell.source.strip() != "":
                    master_nb.cells.append(cell)
    else:
        print(f"Warning: {nb_file} not found.")

output_path = os.path.join(base_path, 'Colab_Master_Pipeline.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(master_nb, f)

print(f"Successfully generated {output_path}")
