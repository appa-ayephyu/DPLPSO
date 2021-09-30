"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces
import numpy as np
import random


class InterventionEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def nCr(self, n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)

    def set_actual_states(self, mental_states, influence):
        self.actual_mental_states = mental_states
        self.actual_influence = influence

    def __init__(self, total_students, sel_students, mu):
        self.total_students = total_students
        self.sel_students = sel_students
        self.mu = mu

        self.VALID_ACTIONS = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        self.students = [i for i in range(self.total_students)]

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds

        low = np.zeros((self.total_students + 1) * self.total_students)
        high = np.empty((self.total_students + 1) * self.total_students)

        for i in range(self.total_students):
            high[i] = self.mu
        for i in range(self.total_students * self.total_students):
            high[i + self.total_students] = 1

        self.action_space = spaces.Discrete(
            self.nCr(self.total_students, self.sel_students)
        )
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.actual_mental_states = np.zeros(self.total_students)
        self.actual_influence = np.zeros((self.total_students, self.total_students))

        # self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def set_state_with_observation(self, obs):
        self.state = obs

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def calculate(self, ind, mental_states, influence, action):
        DELTA = 2
        pos = [i for i in range(self.total_students) if action[i] == 1]
        deltaj = 0
        if ind in pos:
            deltaj = deltaj + DELTA
        for kk in pos:
            numerator = influence[kk][ind] * (self.mu - mental_states[kk]) * DELTA * 1.9
            denominator = influence[kk][ind] * (self.mu - mental_states[kk])
            for neighbors in range(0, self.total_students):
                if neighbors != kk:
                    denominator = (
                        denominator
                        + mental_states[neighbors] * influence[neighbors][ind]
                    )

            result = 0
            if numerator > 0 and denominator > 0:
                result = numerator / denominator
            deltaj = deltaj + round(result)

        return deltaj

    def kthCombination(self, k, l, r):
        # return [1]
        if r == 0:
            return []
        elif len(l) == r:
            return l
        else:
            i = self.nCr(len(l) - 1, r - 1)
            if k < i:
                return l[0:1] + self.kthCombination(k, l[1:], r - 1)
            else:
                return self.kthCombination(k - i, l[1:], r)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        state = self.state

        mental_states = state[0]
        influence = state[1:]

        chosen = self.kthCombination(action, self.students, self.sel_students)

        action_list = [1 if i in chosen else 0 for i in range(self.total_students)]
        # print(action_list)
        pos_selected_students = [
            i for i in range(self.total_students) if action_list[i] == 1
        ]
        # print(pos_selected_students)
        for pos in pos_selected_students:
            mental_states[pos] = self.actual_mental_states[pos]

        updated_mental_states = mental_states

        updated_influence = np.array(
            [
                influence[i][j]
                if self.actual_influence[i][j] == 0
                else self.actual_influence[i][j]
                for i in range(self.total_students)
                for j in range(self.total_students)
            ]
        ).reshape(self.total_students, self.total_students)

        updated_mental_states = [
            updated_mental_states[i]
            - self.calculate(i, mental_states, updated_influence, action_list)
            for i in range(self.total_students)
        ]

        updated_mental_states = [0 if ums < 0 else ums for ums in updated_mental_states]
        self.actual_mental_states = [
            updated_mental_states[i] if i in chosen else self.actual_mental_states[i]
            for i in range(self.total_students)
        ]

        self.state = np.concatenate(
            ([updated_mental_states], updated_influence), axis=0
        )

        done = not np.any(updated_mental_states)

        # print(mental_states)
        # print("updated mental states")
        # print(updated_mental_states)
        if not done:
            reward = np.sum(mental_states) - np.sum(updated_mental_states)
        else:
            reward = 0.0

        # return self.make_image(self.state), reward, done, {}

        return (
            self.state.reshape((self.total_students + 1) * self.total_students),
            reward,
            done,
            {},
        )

    def reset(self):
        self.students = [i for i in range(self.total_students)]
        # mental_states = np.array([[7, 2, 5, 2, 3, 7, 3, 1, 8, 5, 5, 4, 6, 7, 8, 4, 5, 5, 7, 3, 7, 2, 7, 8, 2, 7, 4, 6, 7, 5]]);
        mental_states = np.array(
            [[random.randrange(self.mu + 1) for i in range(self.total_students)]]
        )
        influence = np.array(
            [
                0 if i == j else random.randrange(11) / 10
                for i in range(self.total_students)
                for j in range(self.total_students)
            ]
        ).reshape(self.total_students, self.total_students)
        self.state = np.concatenate((mental_states, influence), axis=0)
        return self.state.reshape((self.total_students + 1) * self.total_students)

    def render(self, mode="human"):
        print("Render: ", self.state)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
