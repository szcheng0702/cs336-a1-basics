import torch
import torch.nn.functional as F


def generate_with_nucleus_sampling(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    """
    Generates text from a prompt using a model and tokenizer with nucleus sampling.

    Args:
        model: A PyTorch model (e.g., from Hugging Face transformers).
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): The initial text to start generation from.
        max_length (int): The maximum length of the generated sequence.
        top_p (float): The cumulative probability for nucleus sampling.
        temperature (float): Controls randomness. Higher values mean more randomness.
    """
    # Set the model to evaluation mode
    model.eval()

    # Get the device the model is on (e.g., 'cuda' or 'cpu')
    device = model.device

    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Autoregressive generation loop
    with torch.no_grad():
        for _ in range(max_length):
            # 1. Get model outputs (logits)
            outputs = model(input_ids)
            # We only need the logits for the last token in the sequence
            next_token_logits = outputs.logits[:, -1, :]

            # 2. Apply temperature scaling
            # A higher temperature flattens the distribution, making less likely tokens more probable
            next_token_logits = next_token_logits / temperature

            # 3. Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # 4. Implement Nucleus Sampling (Top-p filtering)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Create a mask to remove tokens with cumulative probability above the threshold (top_p)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create a mask for the original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )

            # Apply the mask by setting the probability of removed tokens to 0
            probs[indices_to_remove] = 0

            # 5. Renormalize the probabilities
            # This is implicitly handled by torch.multinomial, but explicit renormalization is good practice:
            # probs = probs / probs.sum()

            # 6. Sample the next token from the modified distribution
            next_token_id = torch.multinomial(probs, num_samples=1)

            # 7. Append the new token to the sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # 8. Check for EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # Decode the final sequence of tokens back to text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text
