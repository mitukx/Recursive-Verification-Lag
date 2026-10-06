"""Real two-rank token-level RL update, with no model download for smoke."""
import argparse
import math

from .rvl_systems.lab.distributed_learner import distributed_step, read_samples
from .rvl_systems.types import Generation, VerifiedGeneration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["ddp","fsdp"],default="ddp")
    p.add_argument("--model")
    p.add_argument("--replay")
    p.add_argument("--output",default="artifacts/distributed-lab")
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
    torch.manual_seed(17)
    if a.model:
        if not a.replay:
            p.error("--model requires --replay with behavior token IDs/logprobs")
        model = AutoModelForCausalLM.from_pretrained(a.model)
        samples = read_samples(a.replay)
    else:
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32,n_positions=32,
                                         n_embd=16,n_layer=1,n_head=2,
                                         resid_pdrop=0,embd_pdrop=0,attn_pdrop=0))
        model.eval()
        samples = []
        for i in range(4):
            prompt,response = [1,2],[3+i,7+i]
            with torch.no_grad():
                logits = model(torch.tensor([prompt+response])).logits[0,1:3]
                logps = torch.log_softmax(logits,dim=-1).gather(1,torch.tensor(response)[:,None]).squeeze(1).tolist()
            g = Generation("same-prompt","smoke","token response",sum(logps),2,0,
                           {"prompt_token_ids":prompt,"response_token_ids":response,
                            "response_token_logprobs":logps})
            samples.append(VerifiedGeneration(g,float(i%2),0,0))
    distributed_step(model,samples,a.output,mode=a.mode)


if __name__ == "__main__":
    main()
