import datetime
from tqdm import tqdm

class TxtLogWriter:
    def __init__(self, config: dict, log_txt_path: str):
        self.log_txt_path = log_txt_path
        self.init_setting(config)
    
    def init_setting(self, config: dict):
        self.write_line('Setting:')
        for key, value in config.items():
            self.write_line("\t{} = '{}'".format(key, value))
        self.write_line('')
    def write_metric(self, 
                     identifer: str,
                     epoch: int, 
                     avg_loss: float = None, 
                     avg_acc: float = None):
        if getattr(self, 'is_write_metric', None) == None:
            self.write_line('Progress: ')
        self.is_write_metric = True
        
        if identifer == 'train':
            line = '\tTraining   Epoch {:3d}: avg_accuracy = {:.3f}, avg_loss = {:.3f}'.format(epoch, avg_acc, avg_loss)
        elif identifer == 'validation':
            line = '\tValidation Epoch {:3d}: avg_accuracy = {:.3f}'.format(epoch, avg_acc)
        self.write_line(line)
        
    def write_best_metric(self, 
                        best_epoch: int, 
                        best_accuracy: float):
        self.write_line('\nBest: ')
        self.write_line('\tbest epoch = {:3d}, best_accuracy = {:.3f}'.format(best_epoch, best_accuracy))
        
    def write_line(self, line: str):
        with open(self.log_txt_path, 'a') as f:
            f.write(line + '\n')

def convert_to_scientific(value:float, identifier:str="") -> str:
    """Conver a float value to the string of scientific notation
    """
    if value == None:
        return ""
    else:
        value = '{:.0e}'.format(value)
        if '-' in value:
            index = value.index('-')
        else:
            index = value.index('+')
        if value[index + 1] == '0':
            value = value[:index + 1] + value[index + 2:]
        
        if identifier != "":
            return '&{}{}'.format(identifier, value)
        else:
            return value
        
        
def get_timestamp() -> str:
    utc_now = datetime.datetime.utcnow()
    cur_time = utc_now + datetime.timedelta(hours=8) # Taiwan in UTC+8
    
    timestamp = "[%d-%02d-%02d-%02d%02d]" % (cur_time.year, 
                                            cur_time.month, 
                                            cur_time.day, 
                                            cur_time.hour, 
                                            cur_time.minute)
    return timestamp

def get_progress_bar(identifer: str, 
                     total_steps: int,
                     epoch: int, 
                     num_epochs: int):
    progress_bar = tqdm(total = total_steps, 
                        desc = "{:10s} Epoch {}/{}".format(identifer, epoch+1, num_epochs), 
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    return progress_bar