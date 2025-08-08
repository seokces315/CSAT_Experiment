# Function for mean pooling
def mean_pooling(output, attention_mask):
    # Get last hidden state layer
    last_hidden_state = output.last_hidden_state

    # Prepare mask broadcasting
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden_state)

    # Zero-out & Sum with given hidden states
    sum_embeddings = (last_hidden_state * mask).sum(dim=1)

    # Divide by number of non-masked(non-padded) tokens
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = sum_embeddings / sum_mask

    return embeddings
