import numpy as np

#import config
MAX_FRAMES = 1000

READY_TO_BEGIN_EPISODE = 4
READY_TO_END_EPISODE = 5

class TwoStep():

  def __init__(self, num_trials = 100, init_prob_flag='easy', discount=1., num_steps = 300000,
                envargs = None):
    """Inits the game.

    Args:
      steps_per_ep: default 100
      flip_prob: probability of rerandomizing rewards midway through episode
      init_prob_flag (optional): always 'easy' for this task
      envargs contains all other parameters that might be needed
    """
    
    
    
    
    self._name = "TwoStep"
    self._num_arms = 3
    self._max_reward = 1.0
    self._num_trials = num_trials
    self._num_avg_trials = num_trials
    self._init_prob_flag = init_prob_flag
    self._discount = discount

    # TODO
    #self._flip_prob = envargs.flip_prob
    self._flip_prob = 0.3


    self._prob_transition = np.zeros((1, self._num_arms))   #prob to transition to state 1 (1-p for state 2)
    self._prob_reward = np.zeros((2, self._num_arms))   #prob reward

    self._min_stable = 1   # minimum number of trials before randomizing
    self._size_state = 4
    self._steps_per_trial = 3
    self._steps_per_ep = self._num_trials * self._steps_per_trial
    self._episode = 0

    # added
    self._step = 0
    # case of reset??
    self._total_reward = 0
    self._best_total_reward = 0
    self._best_reward = 0
    self._trial = 0

    #TODO
    #self._reset()
    self.reset()

    #TODO
    self._state = READY_TO_BEGIN_EPISODE

  #TODO
  def reset(self):
    self._init_episode()
    obs = self._current_state
    reward = 0
    done = 0
    reset = done == 1. or self._step == MAX_FRAMES
    # return np.array([[obs]]), reward, done, reset
    return np.array([obs])

  def _init_episode(self):
    """Sets the probabilities of reward for each terminal state
    """
    # get states - one-hots
    self._states = np.zeros((self._size_state, self._size_state))

    # to_ones = np.random.permutation(self._size_state)[0:3]
    for x in xrange(self._size_state):
      # self._states[x][to_ones[x]] = 1
      self._states[x][x] = 1

    self._prob_transition = np.array([[.8,.2]])
    self._randomize()
    self._current_state = 0
    self._last_state = 0
    self._stage = 0
    self._since_flipped = 0

  def _randomize(self):
    rew_probs = [[0.9, 0.9],[0.1, 0.1]]
    self._flipIt = np.random.random(1)[0] < 0.5
    if self._flipIt:
      rew_probs.reverse()
    self._prob_reward = np.array(rew_probs)


  def step(self, actions, agent_id=0):
    """Pulls an arm, returns a reward, whether episode has terminated,
        and best possible reward.

    Args:
      action: Integer between 0 and numActions representation the
          chosen action
    """
    self._last_state = self._current_state

    # TODO
    # action = actions.discrete_actions[0]-1
    action = actions.argmax()

    done = 0
    if self._stage == 0:   # is fixation
      if action == 0:
        reward = 0.
      else:
        reward = -1.
      self._current_state = 1
      self._stage = 1
    elif self._stage == 1:   # is first stage, use prob_transition
      if action == 1 or action == 2:
        if np.random.random() < self._prob_transition[0][action-1]:
          self._current_state = 2
        else:
          self._current_state = 3
        reward = 0.
      else:  # pick a next state at random
        reward = -1.
        self._current_state = np.random.random() < 0.5 and 2 or 3
      self._stage = 2
    else:   # is second stage, use prob_reward
      # Given an action (arm pulled), sample reward, return
      if action == 1 or action == 2:
        current_prob_rewards = self._prob_reward[self._current_state-2]
        self._best_reward = self._max_reward*np.max(current_prob_rewards)
        thisProb = current_prob_rewards[action-1]
        if np.random.random() < thisProb:
          reward = self._max_reward
        else:
          reward = 0.0
      else:
        reward = -1.

      self._total_reward += reward
      self._best_total_reward += self._best_reward
      self._stage = 0
      self._current_state = 0
      self._trial += 1
      self._since_flipped += 1
      # if more than self._min_stable trials since flipping, certain chance of flipping prob rews
      if (self._since_flipped >= self._min_stable) and (np.random.random() <= self._flip_prob):
        self._randomize()
        self._since_flipped = 0


    self._last_action = np.zeros(self._num_arms)
    self._last_action[action] = 1
    # conditions to end episode
    if self._step >= self._steps_per_ep-1:
      self._state = READY_TO_END_EPISODE
      done = 1

    self._step += 1
    self._prev_reward = reward

    obs = self._current_state
    reset = done == 1. or self._step == MAX_FRAMES

    print(np.array([[obs]]).shape)
    return np.array([[obs]]), reward, done, reset
