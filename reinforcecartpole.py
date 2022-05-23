import gym
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import matplotlib.pyplot as plt
from tqdm import tqdm


tf.config.run_functions_eagerly(True)

env = gym.make('CartPole-v0')
env._max_episode_steps = 10000
action_size = env.action_space.n
obs_space = env.observation_space.shape

class ProbabilityDistribution(keras.Model):
	def call(self, inputs):
		return tf.squeeze(tf.random.categorical(inputs, 1), axis=-1)

class Model(keras.Model):
	
	def __init__(self, action_size):
		super().__init__('mlp_policy')
		self.d1 = Dense(128, input_shape=obs_space, activation='relu')
		self.logits = Dense(action_size, activation='softmax')
		self.value = Dense(1, activation='linear')
		self.dist = ProbabilityDistribution()
	
	
	def call(self, inputs):
		x = tf.convert_to_tensor(inputs)
		x = self.d1(x)
		logits = self.logits(x)
		value = self.value(x)
		return logits, value

	def action_value(self, inputs):
		logits, value = self.predict_on_batch(inputs)
		action = self.dist.predict_on_batch(logits)
		return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

model = Model(action_size)
obs = env.reset()
act, value = model.action_value(obs.reshape(1, -1))
print(act, value)
total_episodes = 2000
batch_size = 2
render = False
GAMMA = 0.99
ENTROPY_BETA = 0.1
learning_rate = 0.001

observations = np.empty((batch_size, obs_space[0]))
rewards = np.zeros((batch_size,), dtype=np.int32)
actions = np.empty((batch_size,), dtype=np.int32)
dones = np.empty((batch_size,), dtype=np.int32)

def logits_loss(act_rew, logits):
	actions, rewards = tf.split(act_rew, 2, axis=-1)
	
	log_res = tf.math.log(logits)
	entropy = tf.math.reduce_mean(tf.math.reduce_sum(-(logits * log_res), axis=1))
	if np.isnan(entropy.numpy()):
		print('NOT ENTROPY')
		tf.print(logits)
		tf.print(log_res)
		entropy = tf.constant(0.0)
	entropy_tracker.append(entropy.numpy())
	entropy_loss = -ENTROPY_BETA * entropy

	actions = tf.cast(actions, tf.int32)
	total_losses = rewards * tf.gather(log_res, indices=actions, axis=1, batch_dims=1)
	return keras.backend.mean(total_losses) * -1 + entropy_loss

def value_loss(rewards, values):
	return 0.5 * kls.mean_squared_error(rewards, values)

def calc_qvals(rewards):
	res = []
	sum_r = 0.0
	for r in reversed(rewards):
		sum_r *= GAMMA
		sum_r += r
		res.append(sum_r)
	return np.array(list(reversed(res)))

def get_returns(rewards, dones):
	returns = np.zeros_like(rewards, dtype=np.float32)
	returns[-1] = rewards[-1]
	for t in reversed(range(len(rewards) - 1)):
		if dones[t]:
			rewards[t] = -1
		else:
			returns[t] = rewards[t] + GAMMA * returns[t + 1] * (1 - dones[t])

	return returns

model.compile(optimizer=ko.Adam(lr=learning_rate), loss=[logits_loss, value_loss])
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
	batch_vals = []

	while batch_ep < batch_size:
		batch_states.append(obs.copy())
		action, value = model.action_value(obs.reshape(1, -1))
		batch_actions.append(action)
		batch_vals.append(value)
		try:
			obs, reward, done, _ = env.step(batch_actions[-1])
		except Exception:
			print('EXCEPTION')
			print(batch_actions[-1])
			print(Exception)
			exit(1)
		if render:
			env.render()
		cur_rew.append(reward)
		if done:
			bas_rew = cur_rew
			batch_qvals.extend(calc_qvals(bas_rew))

			smooth_reward.append(np.sum(np.array(cur_rew)))
			epx.append(ep)
			cur_rew.clear()
			batch_ep += 1
			obs = env.reset()


	if ep > 0 and ep % 10 == 0:
		avg_rew.append(np.mean(smooth_reward[-10:]))
		avg_ep.append(ep)
		print("Average Reward:", avg_rew[-1], 'Max reward:', np.max(smooth_reward[-100:]))

	if np.mean(smooth_reward[-100:]) >= 250:
		print('COMPLETE')
		break

	batch_qvals = np.array(batch_qvals)
	batch_vals = np.array(batch_vals)
	batch_vals = np.squeeze(batch_vals)

	# Calculate advanteage
	batch_qvals = batch_qvals - batch_vals

	baseline_tracker.append(np.mean(batch_vals))
	#baseline_tracker.append(baseline)
	#print('baseline', baseline, 'max', np.max(batch_qvals))
	#batch_qvals = batch_qvals - baseline
	
	batch_states = np.array(batch_states)
	batch_actions = np.array(batch_actions)
	act_rew = np.concatenate((batch_actions[:, None], batch_qvals[:, None]), axis=-1)
	losses = model.train_on_batch(batch_states, act_rew)
	loss_tracker.append(losses)

plt.plot(epx, smooth_reward)
#plt.show()
plt.savefig('reward_smooth.png')
plt.clf()

plt.plot(avg_ep, avg_rew)
#plt.show()
plt.savefig('reward_avg.png')
plt.clf()

plt.plot([i for i in range(len(entropy_tracker))], entropy_tracker)
#plt.show()
plt.savefig('entropy.png')
plt.clf()

plt.plot([i for i in range(len(baseline_tracker))], baseline_tracker)
#plt.show()
plt.savefig('baseline.png')
plt.clf()

plt.plot([i for i in range(len(loss_tracker))], loss_tracker)
#plt.show()
plt.savefig('loss.png')
plt.clf()
	

'''
smooth_reward = 0
longest_reward = 0
sm_rewards = []
sm_eps = []
rew_avg = []
avg_eps = []

cur_rew = []
for ep in range(total_episodes):
	tmprewards = []
	tstrewards = []
	batch_rew = []
	for step in range(batch_size):
		observations[step] = obs.copy()
		actions[step] = model.action(obs.reshape(1, -1))
		obs, reward, dones[step], _ = env.step(actions[step])
		rewards[step] = 0
		tstrewards.append(reward)
		tmprewards.append(reward)
		cur_rew.append(reward)

		if render:
			env.render()
		smooth_reward = smooth_reward + 1
		if dones[step]:
			batch_rew.extend(calc_qvals(cur_rew))
			tstrewards[-1] = 0
			cur_rew.clear()
			#tstrewards[-1] = 0
			if smooth_reward > longest_reward:
				#print('LONGEST REWARD', smooth_reward)
				longest_reward = smooth_reward
			sm_rewards.append(smooth_reward)
			sm_eps.append(ep)
			if len(sm_eps) % 100 == 0:
				rew_avg.append(np.mean(np.array(sm_rewards[-100:])))
				avg_eps.append(len(sm_eps))

			smooth_reward = 0
			rewards[step - 1] = np.sum(np.array(tmprewards))
			#print('early', rewards)
			tmprewards = []
			obs = env.reset()
		elif step == batch_size - 1:
			rewards[step] = np.sum(np.array(tmprewards))
			#print('finished', rewards)
	#print(tstrewards)
	
	#print(smooth_reward)
	#returns = get_returns(tstrewards, dones)
	returns = calc_qvals(tstrewards)
	print(tstrewards)
	print(batch_rew)
	#print(returns)
	losses = model.train_on_batch(observations, returns)

plt.plot(sm_eps, sm_rewards)
plt.show()

plt.plot(avg_eps, rew_avg)
plt.show()

'''