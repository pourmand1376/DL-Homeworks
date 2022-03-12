# CE 40719: Deep Learning
## HW6-Q5: REINFORCE (with baseline)

*Full name*:

*STD-ID*: 

In this notebook, you are going to implement REINFORCE algorithm on `CartPole-v0` and compare it to the case with a baseline. To know more about this, please refer to [Sutton&Barto, 13.3-13.4](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf). 


```bash
%%bash
pip install gym pyvirtualdisplay > /dev/null 2>&1
apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
```


```python
import gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

# Auxiliary methods

You can use the following methods to display demos and results of your code.


```python
root = '/content/video'
display = Display(visible=0, size=(200, 150))
display.start()

def show_video(path=root):
    mp4list = glob.glob(f'{path}/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 250px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
    
def wrap_env(env, path=root):
    return Monitor(env, path, force=True)

def plot_curves(curves, title, smooth=True, w_size=50):
    """
    This method plots series specified in `curves['series']`
    inside the same figure.

    - curves: a dictionary, dict(curves=a list of lists, labels=a list of strings);
    - title: figure's title;
    - smooth: whether to take a moving average over each series;
    - w_size: size of the moving average window;

    Notice: Series must have the same length.
    """
    series, labels = curves['series'], curves['labels']  
    N = len(series[0])
    assert all([len(s) == N for s in series])     
    x = list(range(N))
    for s, label in zip(series, labels):
        window = np.ones(w_size)/w_size
        s_smooth = np.convolve(s, window, mode='same')
        y = s_smooth[w_size:N-w_size] if smooth else s
        plt.plot(x[w_size:N-w_size], y, label=label)
    plt.legend()
    plt.title(title)
    plt.show()
```

# `CartPole-v0`

You can see specifications of `CartPole-v0` in the following cell.


```python
env_id = 'CartPole-v0'
env = gym.make(env_id)
spec = gym.spec(env_id)

print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")
print(f"Max Episode Steps: {spec.max_episode_steps}")
print(f"Nondeterministic: {spec.nondeterministic}")
print(f"Reward Range: {env.reward_range}")
print(f"Reward Threshold: {spec.reward_threshold}\n")
```

    Action Space: Discrete(2)
    Observation Space: Box(-3.4028234663852886e+38, 3.4028234663852886e+38, (4,), float32)
    Max Episode Steps: 200
    Nondeterministic: False
    Reward Range: (-inf, inf)
    Reward Threshold: 195.0
    


Here you can see a demo of the completely random policy.


```python
env = gym.make('CartPole-v0')
env = wrap_env(env)
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
env.close()
show_video()
```


<video alt="test" autoplay 
                    loop controls style="height: 250px;">
                    <source src="data:video/mp4;base64,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" type="video/mp4" />
                    </video>


# Method

You can either use `Net` to create baseline and policy networks, or use any other custom architecture. 


```python
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, outdim_policy, outdim_baseline):
        super(Net, self).__init__()
        self.shared = nn.Linear(input_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, outdim_policy)
        self.baseline = nn.Linear(hidden_dim, outdim_baseline)

    def forward(self, x):
        x = self.shared(x)
        x = F.relu(x)
        p = F.softmax(self.policy(x))
        b = self.baseline(x)
        return p, b
```


```python
class REINFORCECartPole():
    def __init__(self, use_baseline=False, GAMMA=None, lr=None):
        self.env_id = 'CartPole-v0'
        self.env = gym.make(self.env_id)
        self.use_baseline = use_baseline
        self.GAMMA = GAMMA
        ############################ ToDo (1 points) #########################
        # Define your network, optimizer, and criterion.
        ######################################################################
        

    def generate_episode(self, video=False, train=True):
        trajectory = []
        ############################ ToDo (2 points) #########################
        # Generate a trajectory from the current policy. This method may be
        # used at training and evaluation time. Also you can record the demo
        # of the trajectory to display later.
        ######################################################################
        
        return trajectory


    def select_action(self, state, train=True):
        ############################ ToDo (4 points) #########################
        # Select action based on `state`. At training time, you should sample
        # from the policy distribution, but at test time, you need to takes
        # the best possible action.
        ######################################################################
        pass

    def train(self, n_episodes, n_eval_episodes=15):
        ############################ ToDo (10 points) ########################
        # Train your networks in the following loop. At the end of each
        # episode, evaluate your networks on `n_eval_episodes` episodes and
        # store average total return of them in `TRs`. You are going to plot
        # these TRs later.
        ######################################################################
        TRs = []
        for i in tqdm(range(n_episodes)):
            pass
        return TRs

    def evaluate(self, n_episodes):
        ############################ ToDo (2 points) #########################
        # Evaluate your networks on `n_episodes` episodes and return the 
        # average **undiscounted** total return.
        ######################################################################
        pass

    def show_demo(self):
        ############################ ToDo (1 points) #########################
        # Display demo of one episode based on the current policy.
        ######################################################################
        pass
```

# Results & conclusion


```python
# First you need to choose appropriate input values.
n = ...
lr = ...
GAMMA = ...
kwargs = dict(GAMMA=GAMMA, lr=lr)
```


```python
reinforce = REINFORCECartPole(**kwargs)
returns_reinforce = reinforce.train(n)
reinforce.show_demo()
```


```python
kwargs['use_baseline'] = True
reinforce_b = REINFORCECartPole(**kwargs)
returns_reinforce_b = reinforce_b.train(n)
reinforce_b.show_demo()
```


```python
############################ ToDo (1 points) ########################
# Plot total return curves for both methods in the same figure.
#####################################################################
```

**Question: (4 points)**

+ Interpret your results. What is the difference between REINFORCE with baseline and without baseline?

+ What is the difference between REINFORCE with baseline and Actor-Critic methods?
