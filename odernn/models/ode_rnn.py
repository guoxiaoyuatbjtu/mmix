import torch.nn as nn
import torch
import models.utils as utils
import numpy as np
from torchdiffeq import odeint as odeint

class ODEFunc(nn.Module):
	def __init__(self, ode_func_net):
		"""
		ode_func_net: neural net that used to transform hidden state in ode
		"""
		super(ODEFunc, self).__init__()
		self.gradient_net = ode_func_net

	def forward(self, t_local, y, backwards = False):
		"""
		Perform one step in solving ODE. Given current data point y and
		current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point
		"""
		grad = self.get_ode_gradient_nn(t_local, y)
		if backwards:
			grad = -grad
		return grad

	def get_ode_gradient_nn(self, t_local, y):
		return self.gradient_net(y)

	def sample_next_point_from_prior(self, t_local, y):
		"""
		t_local: current time point
		y: value at the current time point
		"""
		return self.get_ode_gradient_nn(t_local, y)


class DiffeqSolver(nn.Module):
	def __init__(self, ode_func, method, odeint_rtol=1e-4, odeint_atol=1e-5):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver.
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
			rtol = self.odeint_rtol, atol = self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict):
		"""
		Decode the trajectory through ODE Solver using samples from the prior
		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y
     
class GRU_Unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        utils.init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        utils.init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2))
        utils.init_network_weights(self.new_state_net)


    def forward(self, y_mean, y_std, x, mask):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = torch.chunk(self.new_state_net(concat), chunks=2, dim=-1)
        new_state_std = new_state_std.abs()

        output_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        mask = (torch.sum(mask, -1, keepdim=True) > 0).float()

        new_y = mask * output_y + (1 - mask) * y_mean
        new_y_std = mask * new_y_std + (1 - mask) * y_std

        new_y_std = new_y_std.abs()
        return output_y, new_y, new_y_std


class ODE_RNN(nn.Module):
    """Class for standalone ODE-RNN model. Makes predictions forward in time."""
    def __init__(self, latent_dim, ode_func_layers=1, ode_func_units=100, input_dim=1, decoder_units=100):
        super(ODE_RNN, self).__init__()

        ode_func_net = utils.create_net(latent_dim, latent_dim,
                                        n_layers=ode_func_layers,
                                        n_units=ode_func_units,
                                        nonlinear=nn.Tanh)
        ode_func_net.cuda()

        utils.init_network_weights(ode_func_net)

        rec_ode_func = ODEFunc(ode_func_net=ode_func_net)

        self.ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_units),
            nn.Tanh(),
            nn.Linear(decoder_units, input_dim*2))

        utils.init_network_weights(self.decoder)

        self.gru_unit = GRU_Unit(latent_dim, input_dim, n_units=decoder_units)

        self.latent_dim = latent_dim

        self.sigma_fn = nn.Softplus()

    def forward(self, data, mask, mask_first, time_steps, extrap_time=float('inf'), use_sampling=False):

        batch_size, _, _ = data.size()

        prev_hidden = torch.zeros((batch_size, self.latent_dim))
        prev_hidden_std = torch.zeros((batch_size, self.latent_dim))

        if data.is_cuda:
            prev_hidden = prev_hidden.to(data.get_device())
            prev_hidden_std = prev_hidden_std.to(data.get_device())

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        outputs = []
        prev_observation = data[:, 0]

        if use_sampling:
            prev_output = data[:, 0]

        for i in range(1, len(time_steps)):

            # Make one step.
            if time_steps[i] - time_steps[i - 1] < minimum_step:
                inc = self.ode_solver.ode_func(time_steps[i - 1], prev_hidden)

                ode_sol = prev_hidden + inc * (time_steps[i] - time_steps[i - 1])
                ode_sol = torch.stack((prev_hidden, ode_sol), 1)
            # Several steps.
            else:
                num_intermediate_steps = max(2, ((time_steps[i] - time_steps[i - 1])/minimum_step).int())

                time_points = torch.linspace(time_steps[i - 1], time_steps[i],
                                             num_intermediate_steps)
                ode_sol = self.ode_solver(prev_hidden.unsqueeze(0), time_points)[0]

            hidden_ode = ode_sol[:, -1]

            x_i = prev_observation

            if use_sampling and np.random.uniform(0, 1) < 0.5 and time_steps[i] <= extrap_time:
                x_i = prev_output

            mask_i = mask[:, i]

            output_hidden, hidden, hidden_std = self.gru_unit(hidden_ode, prev_hidden_std,
                                                              x_i, mask_i)

            hidden = mask_first[:, i - 1] * hidden
            hidden_std = mask_first[:, i - 1] * hidden_std

            prev_hidden, prev_hidden_std = hidden, hidden_std

            mean, _ = torch.chunk(self.decoder(output_hidden), chunks=2, dim=-1)

            outputs += [mean]

            if use_sampling:
                prev_output = prev_output*(1 - mask_i) + mask_i*outputs[-1]

            if time_steps[i] <= extrap_time:
                prev_observation = prev_observation*(1 - mask_i) + mask_i*data[:, i]
            else:
                prev_observation = prev_observation*(1 - mask_i) + mask_i*outputs[-1]
        
        outputs.insert(0, data[:,0])
        outputs     = torch.stack(outputs, 1)
        
        return outputs

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
