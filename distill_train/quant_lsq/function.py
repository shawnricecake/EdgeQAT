import torch as t
from .quantizer import IdentityQuan


class QuanLinear(t.nn.Linear):
    def __init__(self, in_features, out_features, bias, quan_w_fn=IdentityQuan(), quan_a_fn=IdentityQuan()):
        super().__init__(in_features, out_features, bias)
        self.quan_w_fn = quan_w_fn
        self.quan_a_fn = quan_a_fn
        self.finish_initial = False

        # xuan: todo: when evaluation, we use this, just uncomment the following is ok
        # self.finish_initial = True
        # self.quan_w_fn.init_from(self.weight)
        # if bias:
        #     self.bias = t.nn.Parameter(self.bias.detach())

    def forward(self, x):
        if not self.finish_initial:
            self.quan_w_fn.init_from(self.weight)
            if self.bias:
                self.bias = t.nn.Parameter(self.bias.detach())
            self.finish_initial = True

        quantized_weight = self.quan_w_fn(self.weight)
        quantized_act = self.quan_a_fn(x)
        try:
            return t.nn.functional.linear(quantized_act, quantized_weight, self.bias)
        except:
            return t.nn.functional.linear(quantized_act, t.transpose(quantized_weight, 0, 1), self.bias)  # for gpt training
