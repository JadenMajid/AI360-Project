import torch


class MuonWithAdam(torch.optim.Optimizer):
    """Native Muon on flattened convolution kernels, Adam on auxiliary weights."""

    def __init__(self, muon_parameters, adam_parameters, config):
        self.original_parameters = [*muon_parameters, *adam_parameters]
        if any(not p.is_contiguous() for p in muon_parameters):
            raise ValueError("Muon convolution kernels must be contiguous")
        self.muon_parameters = muon_parameters
        self.matrices = [p.detach().flatten(1) for p in muon_parameters]
        self.muon = torch.optim.Muon(
            self.matrices, lr=config["learning_rate"], weight_decay=config["weight_decay"],
            momentum=config.get("muon_momentum", 0.95), ns_steps=config.get("muon_ns_steps", 5),
            nesterov=True, adjust_lr_fn=config.get("muon_adjust_lr_fn", "match_rms_adamw"))
        self.adam = torch.optim.Adam(
            adam_parameters, lr=config["learning_rate"], betas=tuple(config["adam_betas"]),
            eps=config["adam_epsilon"], weight_decay=config["weight_decay"])
        super().__init__([*self.muon.param_groups, *self.adam.param_groups], {})
        self.param_groups = [*self.muon.param_groups, *self.adam.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for parameter, matrix in zip(self.muon_parameters, self.matrices):
            matrix.grad = parameter.grad.flatten(1) if parameter.grad is not None else None
        self.muon.step()
        self.adam.step()
        return loss

    def zero_grad(self, set_to_none=True):
        for parameter in self.original_parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()

    def state_dict(self):
        return dict(muon=self.muon.state_dict(), adam=self.adam.state_dict())

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
        self.param_groups = [*self.muon.param_groups, *self.adam.param_groups]
