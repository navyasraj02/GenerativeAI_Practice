from transformers import pipeline

def main():
    # STEP 1 - Instantiate a text-generation pipeline using GPT-2
    generator = pipeline(
        "text-generation",
        model = "gpt2",
        tokenizer = "gpt2",
        device = 0      # GPU = 0, else CPU = -1
    )

    # STEP 2 - Define a prompt you want the model to continue
    prompt = "Once upon a time in a land far, far away,"

    # STEP 3 - Generate text
    outputs = generator(
        prompt,
        max_length = 50,    # total length (prompt + continuation)
        num_return_sequences = 1    # how many different completions to return
    )

    # STEP 4 - Print the generated text
    for i, output in enumerate(outputs):
        generated_text = output["generated_text"]
        print(f"\n--- Generated Sequence {i+1} ---")
        print(generated_text)
        print("-------------------------------\n")

if __name__ == "__main__":
    main()