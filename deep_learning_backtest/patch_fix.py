import nbformat

nb_path = '/home/zombie/Documents/Trading_Agent/deep_learning_backtest/04_dqn_agent_rainbow.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'class DQNAgent:' in cell.source:
            # 1. Rename train_step to learn
            cell.source = cell.source.replace('def train_step(self):', 'def learn(self):')
            
            # 2. Add instantiation if not there
            if 'agent = DQNAgent' not in cell.source:
                # Remove the old print statement
                cell.source = cell.source.replace("print('✅ DQNAgent ready | Rainbow-Lite')", "")
                
                # Append the instantiation block
                instantiation = """
agent = DQNAgent(STATE_SIZE, ACTION_SIZE, cfg)
print(f'✅ DQNAgent ready | ε={agent.epsilon:.3f} | γ={agent.gamma} | Rainbow-Lite')
"""
                cell.source += instantiation

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Fixed agent definition in notebook.")
