import time
import torch

def debug_stream_generate(model, tokenizer, input_ids, max_new_tokens):
    print("[debug] Starting token-by-token generation…")
    model.eval()
    device = next(model.parameters()).device

    generated = input_ids.clone()
    for step in range(max_new_tokens):
        t0 = time.time()

        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)

        decoded = tokenizer.decode(next_token[0])
        dt = time.time() - t0
        print(f"[token {step+1}] {decoded!r} (step time: {dt:.3f}s)")

    return generated


def normal_generate(model, tokenizer, input_ids, attention_mask, **params):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **params
        )
    return outputs
