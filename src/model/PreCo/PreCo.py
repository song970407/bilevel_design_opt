import torch
import torch.nn as nn


class PreCo(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 predictor: nn.Module,
                 corrector: nn.Module,
                 decoder: nn.Module,
                 obs_dim: int = 1):
        super(PreCo, self).__init__()
        self.hidden_dim = hidden_dim
        self.predictor = predictor
        self.corrector = corrector
        self.decoder = decoder
        self.obs_dim = obs_dim

    def correct(self, g, h, x):
        return self.corrector(g, h, x)

    def predict(self, g, h, u):
        return self.predictor(g, h, u)

    def filter_history(self, g, history_xs, history_us, h0=None):
        """ Estimate hidden from the sequences of the observations and controls.
                Args:
                    g:
                    history_xs: [#.total state nodes x history_len x state_dim]
                    history_us: [#.total control nodes x (history_len-1) x control_dim]
                    h0: initial hidden [#.total state nodes x hidden_dim]
                Returns:
                """
        if h0 is None:
            h = torch.zeros(g.number_of_nodes(), self.hidden_dim).to(history_xs.device)
        else:
            h = h0

        history_xs = history_xs.unbind(dim=1)
        history_us = history_us.unbind(dim=1)

        for i in range(len(history_us)):
            h = self.correct(g, h, history_xs[i])  # correct hidden
            h = self.predict(g, h, history_us[i])  # predict next hidden
        h = self.correct(g, h, history_xs[-1])
        return h

    def multi_step_prediction(self, g, h, us):
        """
        Args:
            g:
            h: filtered hidden [#.total state nodes x hidden_dim]
            us: [#.total control nodes x prediction_length x control_dim]
        Returns:
        """
        hs = []
        us = us.unbind(dim=1)
        for u in us:
            h = self.predict(g, h, u)
            hs.append(h)
        hs = torch.stack(hs, dim=1)  # [#. total state nodes x prediction_length x  hidden_dim]
        xs = self.decoder(hs)  # [#. total state nodes x prediction_length x  state_dim]
        return xs

    def rollout(self, g, hc, us, ys):
        """
        Args:
            g: graph
            hc: initial hidden
            us: u_t ~ u_t+(k-1) # [#. total control nodes x rollout length x control dim]
            ys: x_t+1 ~ x_t+k
        Returns:
        """
        K = us.shape[1]
        us = us.unbind(dim=1)
        ys = ys.unbind(dim=1)

        hps = []  # predicted hiddens
        hcs = []  # corrected hiddens

        # performs one-step prediction recursively.
        for k in range(K):
            hp = self.predict(g, hc, us[k])
            hc = self.correct(g, hp, ys[k])
            hps.append(hp)
            hcs.append(hc)

        # one-step prediction results
        # hcs = [hc_t+1, hc_t+2, ..., hc_t+k]
        hcs = torch.stack(hcs, dim=1)  # [#. total state nodes x rollout length x hidden_dim]

        # hps = [hp_t+1, hp_t+2, ..., hp_t+k]
        hps = torch.stack(hps, dim=1)  # [#. total state nodes x rollout length x hidden_dim]

        # performs latent overshooting
        latent_overshoot_hps = torch.zeros(g.number_of_nodes(), K, K, self.hidden_dim).to(us[0].device)
        latent_overshoot_mask = torch.zeros(g.number_of_nodes(), K, K, self.obs_dim).to(us[0].device)
        for i, hp in enumerate(hps.unbind(dim=1)[:-1]):
            latent_hps = []
            for j in range(i + 1, K):
                hp = self.predict(g, hp, us[j])
                latent_hps.append(hp)
            latent_hps = torch.stack(latent_hps, dim=1)
            latent_overshoot_hps[:, i, i + 1:, :] = latent_hps
            latent_overshoot_mask[:, i, i + 1:, :] = 1.0

        # decoding the one-step prediction results
        hcs_dec = self.decoder(hcs)  # [x_t+1, ..., x_t+k]
        hps_dec = self.decoder(hps)  # [x_t+1, ..., x_t+k]

        # latent the latent overshooting results
        latent_overshoot_dec = self.decoder(latent_overshoot_hps)

        ret = dict()
        ret['hcs_dec'] = hcs_dec
        ret['hps_dec'] = hps_dec
        ret['latent_overshoot_dec'] = latent_overshoot_dec
        ret['latent_overshoot_mask'] = latent_overshoot_mask
        return ret


class HeteroPreCo(PreCo):
    def __init__(self,
                 hidden_dim: int,
                 predictor: nn.Module,
                 corrector: nn.Module,
                 decoder: nn.Module,
                 obs_dim: int = 1):
        super(HeteroPreCo, self).__init__(hidden_dim, predictor, corrector, decoder, obs_dim)

    def filter_history(self, g, history_xs, history_us, h0=None):
        """
        Estimate hidden from the sequences of the observations and controls.
        :param g: graph
        :param history_xs: [#.total state nodes x history_len x state_dim]
        :param history_us: [#.total control nodes x (history_len-1) x control_dim]
        :param h0: initial hidden [#.total state nodes x hidden_dim]
        :return: filtered history h, [#.total state nodes x hidden_dim]
        """
        if h0 is None:
            h = torch.zeros(g.number_of_nodes(ntype='state'), self.hidden_dim).to(history_xs.device)
        else:
            h = h0

        history_xs = history_xs.unbind(dim=1)
        history_us = history_us.unbind(dim=1)

        for i in range(len(history_us)):
            h = self.correct(g, h, history_xs[i])  # correct hidden
            h = self.predict(g, h, history_us[i])  # predict next hidden
        h = self.correct(g, h, history_xs[-1])
        return h

    def multi_step_prediction(self, g, h, us):
        """
        :param g: graph
        :param h: filtered hidden [#.total state nodes x hidden_dim]
        :param us: [#.total control nodes x prediction_length x control_dim]
        :return:
        """
        hs = []
        us = us.unbind(dim=1)
        for u in us:
            h = self.predict(g, h, u)
            hs.append(h)
        hs = torch.stack(hs, dim=1)  # [#. total state nodes x prediction_length x  hidden_dim]
        xs = self.decoder(hs)  # [#. total state nodes x prediction_length x  state_dim]
        return xs

    def rollout(self, g, hc, us, ys):
        """
        :param g: graph
        :param hc: filtered hidden [#.total state nodes x hidden_dim]
        :param us: u_t ~ u_t+(k-1)  [#. total control nodes x rollout length x control dim]
        :param ys: x_t+1 ~ x_t+k [#. total state_nodes x rollout length x state_dim]
        :return:
        """
        K = us.shape[1]
        ys = ys.unbind(dim=1)
        us = us.unbind(dim=1)

        hps = []  # predicted hiddens
        hcs = []  # corrected hiddens

        # performs one-step prediction recursively.
        for k in range(K):
            hp = self.predict(g, hc, us[k])
            hc = self.correct(g, hp, ys[k])
            hps.append(hp)
            hcs.append(hc)

        # one-step prediction results
        # hcs = [hc_t+1, hc_t+2, ..., hc_t+k]
        hcs = torch.stack(hcs, dim=1)  # [#. total state nodes x rollout length x hidden_dim]

        # hps = [hp_t+1, hp_t+2, ..., hp_t+k]
        hps = torch.stack(hps, dim=1)  # [#. total state nodes x rollout length x hidden_dim]

        # performs latent overshooting
        latent_overshoot_hps = torch.zeros(g.number_of_nodes(ntype='state'), K, K, self.hidden_dim).to(us[0].device)
        latent_overshoot_mask = torch.zeros(g.number_of_nodes(ntype='state'), K, K, self.obs_dim).to(us[0].device)
        for i, hp in enumerate(hps.unbind(dim=1)[:-1]):
            latent_hps = []
            for j in range(i + 1, K):
                hp = self.predict(g, hp, us[j])
                latent_hps.append(hp)
            latent_hps = torch.stack(latent_hps, dim=1)
            latent_overshoot_hps[:, i, i + 1:, :] = latent_hps
            latent_overshoot_mask[:, i, i + 1:, :] = 1.0

        # decoding the one-step prediction results
        hcs_dec = self.decoder(hcs)  # [x_t+1, ..., x_t+k]
        hps_dec = self.decoder(hps)  # [x_t+1, ..., x_t+k]

        # latent the latent overshooting results
        latent_overshoot_dec = self.decoder(latent_overshoot_hps)

        ret = dict()
        ret['hcs_dec'] = hcs_dec
        ret['hps_dec'] = hps_dec
        ret['latent_overshoot_dec'] = latent_overshoot_dec
        ret['latent_overshoot_mask'] = latent_overshoot_mask
        return ret
