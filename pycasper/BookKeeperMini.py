import os
from pathlib import Path

class BookKeeperMini:
    def __init__(
        self,
        project_name,
        save_dir="save",
    ):
        self.save_dir = save_dir
        self.project_name = project_name
        
        self.exp_filename = '.experiments'
        self.init_exp()

    def init_exp(self):
        if not Path(self.exp_filename).exists():
            self.write_exp(0)
            
    def write_exp(self, exp):
        with open(self.exp_filename, 'w') as f:
            f.writelines(['{}'.format(exp)])
        
    def get_exp(self):
        with open(self.exp_filename, 'r') as f:
            line = f.readline()
        return int(line)
    
    def new_exp(self):
        self.exp_num = self.get_exp() + 1
        self.write_exp(self.exp_num)
        self.save_path = Path(self.save_dir)/self.project_name/'exp_{:03d}'.format(self.exp_num)
        os.makedirs(self.save_path, exist_ok=True)
