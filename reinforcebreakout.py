import gym
import numpy as np
import collections
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow.keras.optimizers as ko
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make('Breakout-v0')
action_size = env.action_space.n
obs_space = env.observation_space.shape

tf.config.run_functions_eagerly(True)

def process_frame(frame):
	gray = np.mean(frame, axis=2)
	norm = gray / 255.
	crop = norm[50:-10, 5:-5]
	crop = crop[::2, ::2]
	return crop

stack_size = 4

stacked_frames = collections.deque(maxlen=stack_size)
def stack_frames(state, is_new = False):
	state = process_frame(state)
	if is_new:
		for _ in range(stack_size):
			stacked_frames.append(state)
	else:
		stacked_frames.append(state)
	return np.stack(stacked_frames, axis=2)

class ProbabilityDistribution(keras.Model):
    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(keras.Model):

	def __init__(self, action_size):
		super().__init__('mlp_policy')
		self.conv1 = Conv2D(32, 8, strides=4, activation='relu')
		self.conv2 = Conv2D(64, 4, strides=2, activation='relu')
		self.conv3 = Conv2D(64, 3, strides=1, activation='relu')
		self.fl = Flatten()
		self.h = Dense(512, activation='relu')
		self.logits = Dense(action_size, activation='softmax')
		self.dist = ProbabilityDistribution()

	def call(self, inputs):
		x = tf.convert_to_tensor(inputs)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.fl(x)
		x = self.h(x)
		return self.logits(x)

	def action_value(self, inputs):
		logits = self.predict_on_batch(inputs)
		action = self.dist.predict_on_batch(logits)
		return np.squeeze(action, axis=-1)

model = Model(action_size)
obs = env.reset()
obs = stack_frames(obs, is_new=True)

total_episodes = 300
batch_size = 1
render = False
GAMMA = 0.99
learning_rate = 0.001
ENTROPY_BETA = 0.1

def calc_loss(act_rew, logits):
	actions, rewards = tf.split(act_rew, 2, axis=-1)
	
	log_res = tf.math.log(logits)
	entropy = tf.math.reduce_mean(tf.math.reduce_sum(-(logits * log_res), axis=1))
	entropy_loss = -ENTROPY_BETA * entropy
	entropy_tracker.append(entropy.numpy())

	actions = tf.cast(actions, tf.int32)
	total_losses = rewards * tf.gather(log_res, indices=actions, axis=1, batch_dims=1)
	tf.print(keras.backend.mean(total_losses) * -1, keras.backend.mean(total_losses) * -1 + entropy_loss)
	return keras.backend.mean(total_losses) * -1 + entropy_loss

def calc_qvals(rewards):
	res = []
	sum_r = 0.0
	for r in reversed(rewards):
		sum_r *= GAMMA
		sum_r += r
		res.append(sum_r)
	return np.array(list(reversed(res)))

model.compile(optimizer=ko.Adam(lr=learning_rate), loss=calc_loss)
smooth_reward = []
avg_rew = []
avg_ep = []
epx = []
entropy_tracker = []
baseline_tracker = []
loss_tracker = []

for ep in tqdm(range(total_episodes)):
	batch_ep = 0
	batch_states = []
	batch_actions = []
	cur_rew = []
	batch_qvals = []
	step_idx = 0

	while batch_ep < batch_size:
		batch_states.append(obs.copy())
		action = model.action_value(obs.reshape(1, *obs.shape))
		batch_actions.append(action)
		obs, reward, done, _ = env.step(batch_actions[-1])
		if render:
			env.render()
		cur_rew.append(reward)
		obs = stack_frames(obs)

		if done:
			baseline = np.array(0.0)
			if smooth_reward:
				baseline = np.mean(smooth_reward) / 2
			baseline_tracker.append(baseline)
			batch_qvals.extend(calc_qvals(cur_rew))
			smooth_reward.append(np.sum(np.array(cur_rew)))
			epx.append(ep)
			cur_rew.clear()
			batch_ep += 1
			obs = stack_frames(env.reset(), is_new=True)

			if ep > 0 and ep % 100 == 0:
				avg_rew.append(np.mean(smooth_reward[-100:]))
				avg_ep.append(ep)
				print("Average Reward:", avg_rew[-1])

	batch_qvals = np.array(batch_qvals)# - baseline
	batch_states = np.array(batch_states)
	batch_actions = np.array(batch_actions)
	act_rew = np.concatenate((batch_actions[:, None], batch_qvals[:, None]), axis=-1)
	losses = model.train_on_batch(batch_states, act_rew)
	loss_tracker.append(losses)


plt.plot(epx, smooth_reward)
plt.show()

plt.plot(avg_ep, avg_rew)
plt.show()

plt.plot([i for i in range(len(entropy_tracker))], entropy_tracker)
plt.show()

plt.plot(epx, baseline_tracker)
plt.show()

plt.plot([i for i in range(len(loss_tracker))], loss_tracker)
plt.show()	