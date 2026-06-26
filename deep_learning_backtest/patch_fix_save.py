import nbformat

nb_path = '/home/zombie/Documents/Trading_Agent/deep_learning_backtest/04_dqn_agent_rainbow.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'class DQNAgent:' in cell.source:
            if 'def save(self, path):' not in cell.source:
                # Add save and load before the agent instantiation
                methods = """
    def save(self, path):
        import torch
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
            'update_step': self.update_step,
        }, path)

    def load(self, path):
        import torch
        ckpt = torch.load(path, map_location=device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon     = ckpt['epsilon']
        self.update_step = ckpt['update_step']
"""
                # Find the line 'agent = DQNAgent' and insert methods right before it
                parts = cell.source.split('agent = DQNAgent')
                cell.source = parts[0] + methods + '\nagent = DQNAgent' + parts[1]

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Added save and load methods.")
