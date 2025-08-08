from transformers import AutoModel, AutoTokenizer


# Function to load tokenizer & model
def load_model(model_id):
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    return tokenizer, model
