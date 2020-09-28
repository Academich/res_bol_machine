import numpy as np
import torch


class RBM:

    def __init__(self, n_visible, n_hidden, k, n_epochs, batch_size):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_gibbs_sampling_steps = k
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.weights = torch.randn(n_visible, n_hidden) * 0.1  # from standard normal distribution
        self.bias_visible = torch.ones(n_visible) * 0.5
        self.bias_hidden = torch.zeros(n_hidden)

        # self.learning_rate = None
        # self.momentum_coef = None
        # self.weight_decay = None
        # self.weights_momentum = torch.zeros(n_visible, n_hidden)
        # self.visible_bias_momentum = torch.zeros(n_visible)
        # self.hidden_bias_momentum = torch.zeros(n_hidden)

    def calc_probs_of_hidden(self, visible):
        logits_of_hidden = torch.matmul(visible, self.weights) + self.bias_hidden
        probs_of_hidden = torch.sigmoid(logits_of_hidden)
        return probs_of_hidden

    def calc_probs_of_visible(self, hidden):
        logits_of_visible = torch.matmul(hidden, self.weights.t()) + self.bias_visible
        probs_of_visible = torch.sigmoid(logits_of_visible)
        return probs_of_visible

    def draw_values_from_probs(self, probs):
        return torch.bernoulli(probs)

    def visible_from_gibbs_sampling(self, initial_visible: torch.Tensor):
        visible_at_step = initial_visible
        for _ in range(self.n_gibbs_sampling_steps):
            probs_hidden_at_step = self.calc_probs_of_hidden(visible_at_step)
            hidden_at_step = self.draw_values_from_probs(probs_hidden_at_step)
            probs_visible_at_step = self.calc_probs_of_visible(hidden_at_step)
            visible_at_step = self.draw_values_from_probs(probs_visible_at_step)

            visible_at_step[initial_visible < 0] = initial_visible[initial_visible < 0]

        final_visible = visible_at_step
        return final_visible

    def contrastive_divergence_step(self, init_vis, init_probs_hid, fin_vis, fin_probs_hid):
        positive_associations = torch.mm(init_vis.t(), init_probs_hid)
        negative_associations = torch.mm(fin_vis.t(), fin_probs_hid)
        self.weights += (positive_associations - negative_associations)
        self.bias_visible += torch.sum((init_vis - fin_vis), 0)
        self.bias_hidden += torch.sum((init_probs_hid - fin_probs_hid), 0)

    def train(self, data: torch.Tensor):
        for epoch in range(self.n_epochs):
            epoch_error = []
            for i in range(0, data.size(0) - self.batch_size, self.batch_size):
                init_visible = data[i:i + self.batch_size]
                init_probs_hid = self.calc_probs_of_hidden(init_visible)
                gibbs_spl_visible = self.visible_from_gibbs_sampling(init_visible)
                gibbs_spl_probs_hid = self.calc_probs_of_hidden(gibbs_spl_visible)
                self.contrastive_divergence_step(init_visible,
                                                 init_probs_hid,
                                                 gibbs_spl_visible,
                                                 gibbs_spl_probs_hid)
                batch_error = torch.sum(
                    torch.abs(init_visible[init_visible >= 0] - gibbs_spl_visible[init_visible >= 0]))
                epoch_error += [batch_error]
            mean_n_errors_epoch = np.mean(epoch_error)
            print(f'Epoch: {epoch}. Mean number of errors: {mean_n_errors_epoch}')

    def evaluate(self, data: torch.Tensor):
        hidden_features_probs = self.calc_probs_of_hidden(data)
        hidden_features_vals = self.draw_values_from_probs(hidden_features_probs)
        reconstruction_probs = self.calc_probs_of_visible(hidden_features_vals)
        reconstruction = self.draw_values_from_probs(reconstruction_probs)
        reconstruction_errors = torch.sum(torch.abs(data[data >= 0] - reconstruction[data >= 0])).int().item()
        new_answers = data[data < 0].numel() - data[reconstruction < 0].numel()
        print("Reconstruction errors: ", reconstruction_errors)
        print("New answers: ", new_answers)
        return reconstruction
