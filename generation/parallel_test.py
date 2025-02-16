from accelerate import Accelerator
import torch
from diffusers import FluxPipeline  # your pipeline class

# Initialize Accelerator, which sets up a process group if needed
accelerator = Accelerator()

# Load your pipeline (it should already use device_map="balanced" if needed)
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=None,
    vae=None,
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    max_memory={0: "24GB", 1: "24GB"},
)
# pipeline.to(accelerator.device)  # move the pipeline to the local device

# Assume you have a list of prompts you want to process as a batch:
prompts = ["photo of a dog", "photo of a cat", "photo of a bird", "photo of a fish"]

# The accelerator provides a convenient split utility:
with accelerator.split_between_processes(prompts) as prompt_subset:
    # Each process now gets a subset of the prompts.
    # Encode your prompt(s)
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt_subset, prompt_2=None, max_sequence_length=512
    )

    # Run the denoising process for this sub-batch
    latents = pipeline(
        prompt_embeds=prompt_embeds,
        output_type="latent",
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=20,
        guidance_scale=0,
        height=1024, width=1024,
        generator=torch.Generator("cuda").manual_seed(1000),
    ).images

    # Optionally, save or gather your results
    for idx, latent in enumerate(latents):
        latent.save(f"result_{accelerator.process_index}_{idx}.jpg")
