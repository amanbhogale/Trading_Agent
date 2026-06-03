import json

file_path = "/home/zombie/Documents/Trading_Agent/deep_learning_backtest/03_lstm_model.ipynb"
with open(file_path, "r") as f:
    notebook = json.load(f)

code = []
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        for line in cell["source"]:
            code.append(line if line.endswith('\n') else line + '\n')

with open("test_script.py", "w") as f:
    f.writelines(code)
