import numpy as np
import abc

class AbstractReplayBuffer(abc.ABC):
   @abc.abstractmethod
   def add(self, time_step):
       pass

   @abc.abstractmethod
   def __next__(self, ):
       pass

   @abc.abstractmethod
   def __len__(self, ):
       pass


class EfficientReplayBuffer(AbstractReplayBuffer):
    def __init__(self, obs_shape, act_shape, buffer_size, 
                 batch_size, nstep, discount, frame_stack,
                 data_specs=None):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        # fixed since we can only sample transitions that occur nstep earlier
        # than the end of each episode or the last recorded observation
        self.discount_vec = np.power(discount, np.arange(nstep)).astype('float32')
        self.next_dis = discount**nstep
        
        self.obs_shape = obs_shape
        self.act_shape = act_shape

    def _initial_setup(self):
        self.index = 0
        
        self.ims_channels = self.obs_shape[2] // self.frame_stack

        self.obs = np.zeros([self.buffer_size, *self.obs_shape[:2], self.ims_channels], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        # which timesteps can be validly sampled (Not within nstep from end of
        # an episode or last recorded observation)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, time_step, first):
        
        if first:
            latest_obs = time_step[0,:,:,-self.ims_channels:]
            # if first observation in a trajectory, record frame_stack copies of it
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            
            latest_obs = time_step[0]
            action = time_step[1]
            reward = time_step[2]
            discount = time_step[3]
            
            np.copyto(self.obs[self.index], latest_obs)
            np.copyto(self.act[self.index], action)
            self.rew[self.index] = reward
            self.dis[self.index] = discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step, first=False):
        # time_step_format: (observation, action, reward, discount)
        # if first=True: (observation)
        if self.index == -1:
            self._initial_setup()
        self.add_data_point(time_step, first)

    def __next__(self, ):
        # sample only valid indices
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:] # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement reward computation as a matmul in pytorch for
        # marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        ret = (obs, act, rew, dis, nobs)
        return ret

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index
