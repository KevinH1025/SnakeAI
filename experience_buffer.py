import random
import pickle
from collections import deque 

class MemoryBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size) # when the length is reached it automatically pops right

    # add each snake step to the buffer
    def add(self, state, action, reward, next_state, dead):
        self.buffer.append((state, action, reward, next_state, dead))

    # returns a random batch of samples if bigger than the specified batch size
    def get_batch(self, batch_size=1000):
        if len(self.buffer) < batch_size:
            return self.buffer
        else:
            return random.sample(self.buffer, batch_size)
        
    def save_buffer(self, filename="model/replay_buffer.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.buffer, f)
        print(f"Replay buffer saved to {filename}")
    
    def load_buffer(self):
        pass