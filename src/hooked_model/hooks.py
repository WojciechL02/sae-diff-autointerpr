import numpy as np
import torch


class SAEReconstructHook:
    def __init__(
        self,
        sae,
    ):
        self.sae = sae

    @torch.no_grad()
    def __call__(self, module, input, output):
        output1, output2 = output[0].chunk(2)
        # reshape to SAE input shape
        output1 = output1.permute(0, 2, 3, 1).reshape(
            len(output1), output1.shape[-1] * output1.shape[-2], -1
        )
        output2 = output2.permute(0, 2, 3, 1).reshape(
            len(output2), output2.shape[-1] * output2.shape[-2], -1
        )
        output_cat = torch.cat([output1, output2], dim=0)
        sae_input, _, _ = self.sae.preprocess_input(output_cat)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        sae_out = (latents @ self.sae.W_dec) + self.sae.b_dec
        sae_out1 = sae_out[: output1.shape[1] * len(output1)]
        sae_out2 = sae_out[output1.shape[1] * len(output1) :]
        hook_output = torch.cat(
            [
                sae_out1.reshape(
                    len(output1),
                    int(np.sqrt(output1.shape[-2])),
                    int(np.sqrt(output1.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
                sae_out2.reshape(
                    len(output2),
                    int(np.sqrt(output2.shape[-2])),
                    int(np.sqrt(output2.shape[-2])),
                    -1,
                ).permute(0, 3, 1, 2),
            ],
            dim=0,
        )

        return (hook_output,)


class AblateHook:
    @torch.no_grad()
    def __call__(self, module, input, output):
        if isinstance(input, tuple):
            return (input[0],)
        return input[0]
