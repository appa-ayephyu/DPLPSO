import pyswarms as ps
from interventions import InterventionEnv


class DPSO_Intervention:
    def __init__(
        self,
        state,
        total_students,
        sel_students,
        mu,
        actual_mental_states,
        actual_influence,
    ):

        self.total_students = total_students
        self.state = state
        self.actual_mental_states = actual_mental_states
        self.actual_influence = actual_influence
        self.env = InterventionEnv(
            total_students=total_students, sel_students=sel_students, mu=mu
        )
        self.env.set_actual_states(self.actual_mental_states, self.actual_influence)

    def step_dpso(self, actions):
        actions = [x[0] for x in actions]
        reward_list = []
        for action in actions:
            self.env.set_state_with_observation(self.state)
            self.env.set_actual_states(self.actual_mental_states, self.actual_influence)
            state_r, reward, done_r, dummy = self.env.step(action)
            reward_list.append(-1 * reward)
        return reward_list

    def optimize_action(self):
        # Set-up hyperparameters
        options = {"c1": 0.5, "c2": 0.3, "k": 2, "p": 2, "w": 0.9}

        # Call instance of PSO
        optimizer = ps.discrete.binary.BinaryPSO(
            n_particles=10, dimensions=self.total_students, options=options
        )

        # Perform optimization
        cost, pos = optimizer.optimize(self.step_dpso, iters=100)

        return pos
