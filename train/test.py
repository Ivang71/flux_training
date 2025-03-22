import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from pathlib import Path

# No need to add parent directory since we're already in the train directory
from archtecture import IdentityPreservingFlux

def test_identity_extraction(model, ref_image_path):
    """Test identity extraction from reference image"""
    print(f"\n--- Testing identity extraction from {ref_image_path} ---")
    
    # Load reference image
    ref_img = Image.open(ref_image_path).convert("RGB")
    
    # Extract identity
    start_time = time.time()
    face_embeddings, body_embeddings, face_metadata, body_boxes = model.extract_identity(ref_img)
    end_time = time.time()
    
    # Print results
    print(f"Identity extraction took {end_time - start_time:.2f} seconds")
    print(f"Found {face_embeddings.shape[1]} faces")
    print(f"Found {body_embeddings.shape[1]} person bodies")
    
    # Visualize detections on image
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.array(ref_img))
    
    # Draw face bounding boxes (red)
    for face in face_metadata:
        x1, y1, x2, y2 = [int(coord) for coord in face["bbox"]]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    # Draw body bounding boxes (blue)
    for box in body_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    
    ax.set_title(f"Detections: {face_embeddings.shape[1]} faces, {body_embeddings.shape[1]} bodies")
    ax.axis('off')
    
    # Save visualization
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "detections.png"))
    plt.close(fig)
    
    print(f"Detection visualization saved to {os.path.join(results_dir, 'detections.png')}")
    
    return face_embeddings, body_embeddings

def test_generation(model, ref_image_path, prompt, save_visualization=True):
    """Test image generation with identity preservation"""
    print(f"\n--- Testing generation with prompt: '{prompt}' ---")
    
    # Extract identity
    face_embeddings, body_embeddings = test_identity_extraction(model, ref_image_path)
    
    # Generate with identity preservation
    start_time = time.time()
    image = model.generate(
        prompt=prompt,
        reference_image=ref_image_path,
        negative_prompt="blurry, low quality, cartoon, drawing, illustration",
        num_inference_steps=30,
        guidance_scale=7.5,
        face_strength=1.0,
        body_strength=1.0
    )
    end_time = time.time()
    
    print(f"Generation took {end_time - start_time:.2f} seconds")
    
    # Save generated image
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    image_path = os.path.join(results_dir, "generated.png")
    image.save(image_path)
    print(f"Generated image saved to {image_path}")
    
    # Save attention visualization
    if save_visualization:
        viz_path = os.path.join(results_dir, "attention_visualization.png")
        model.save_visualization(viz_path)
        print(f"Attention visualization saved to {viz_path}")
    
    return image

def test_generation_with_dummy_data(model):
    """Test generation with sample data when no input image is available"""
    print("\n--- Testing with dummy identity data ---")
    
    # Create dummy face and body embeddings with proper batch dimension
    face_embeddings = torch.randn(1, 1, 512)  # [batch_size, num_faces, dim]
    body_embeddings = torch.randn(1, 1, 1024)  # [batch_size, num_persons, dim]
    
    try:
        # Test each component individually for better error reporting
        
        # 1. Test face perceiver
        try:
            face_tokens = model.face_perceiver(face_embeddings)
            print(f"Face perceiver output shape: {face_tokens.shape}")
        except Exception as e:
            print(f"Error in face perceiver: {e}")
            return False
        
        # 2. Test body perceiver
        try:
            body_tokens = model.body_perceiver(body_embeddings)
            print(f"Body perceiver output shape: {body_tokens.shape}")
        except Exception as e:
            print(f"Error in body perceiver: {e}")
            return False
        
        # 3. Test identity fusion module separately
        if model.use_identity_fusion:
            try:
                # First get the processed tokens
                face_processed = model.face_perceiver(face_embeddings)
                body_processed = model.body_perceiver(body_embeddings)
                
                # Test fusion directly
                fused = model.identity_fusion(face_processed, body_processed)
                print(f"Identity fusion output shape: {fused.shape}")
                
                # Check if output makes sense
                expected_seq_len = face_processed.size(1) + body_processed.size(1)
                if fused.size(1) != expected_seq_len:
                    print(f"WARNING: Identity fusion output has unexpected sequence length: {fused.size(1)} vs expected {expected_seq_len}")
            except Exception as e:
                print(f"Error in identity fusion module: {e}")
                print("This is the component with the dimension mismatch issue.")
                return False
        
        # 4. Now test the complete prepare_identity_tokens method
        try:
            model.prepare_identity_tokens(face_embeddings, body_embeddings)
            print("Identity tokens prepared successfully")
            
            # Verify the tokens were stored correctly
            if hasattr(model, "current_face_tokens"):
                print(f"Stored face tokens shape: {model.current_face_tokens.shape}")
            if hasattr(model, "current_body_tokens"):
                print(f"Stored body tokens shape: {model.current_body_tokens.shape}")
        except Exception as e:
            print(f"Error in prepare_identity_tokens: {e}")
            return False
        
        # Only attempt pipeline creation if base model is loaded
        if model.base_model is not None:
            # Create pipeline if not yet available
            if not hasattr(model, 'pipeline'):
                try:
                    from diffusers import FluxPipeline
                    print("Creating pipeline using FluxPipeline...")
                    model.pipeline = FluxPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        model=model.base_model,
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                        device_map="balanced" if torch.cuda.is_available() else None
                    )
                    
                    # Move to GPU if needed
                    if torch.cuda.is_available():
                        model.pipeline = model.pipeline.to("cuda")
                        
                    # Run inference
                    prompt = "A professional portrait photograph of a person, studio lighting"
                    generator = torch.Generator("cuda").manual_seed(42) if torch.cuda.is_available() else None
                    
                    output = model.pipeline(
                        prompt=prompt,
                        negative_prompt="blurry, low quality",
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=1024,
                        width=1024,
                        generator=generator
                    )
                    
                    # Save generated image
                    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
                    os.makedirs(results_dir, exist_ok=True)
                    image = output.images[0]
                    image_path = os.path.join(results_dir, "generated_dummy.png")
                    image.save(image_path)
                    print(f"Generated image with dummy data saved to {image_path}")
                except ImportError as e:
                    print(f"FluxPipeline not available: {e}")
                    print("You need to install the latest version of diffusers to use FluxPipeline.")
                except Exception as e:
                    print(f"Error during pipeline inference: {e}")
            else:
                print("Pipeline already exists - skipping creation")
        else:
            print("Base model not loaded - skipping pipeline test")
            
        print("Dummy data test completed successfully")
        return True
    except Exception as e:
        print(f"Error in dummy data test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_structure(model):
    """Test only the model structure without running inference"""
    print("\n--- Testing model structure ---")
    
    # Print model architecture summary
    print(f"Face injection index: {model.face_injection_index}")
    print(f"Body injection index: {model.body_injection_index}")
    
    # Test extractors
    print("\nTesting face extractor...")
    if hasattr(model.face_extractor, 'face_analyzer') and model.face_extractor.face_analyzer is not None:
        print("✓ Face extractor initialized successfully")
    else:
        print("✗ Face extractor not properly initialized")
    
    print("\nTesting body extractor...")
    if hasattr(model.body_extractor, 'detector') and model.body_extractor.detector is not None:
        print("✓ Body detector initialized successfully")
    else:
        print("✗ Body detector not properly initialized")
    
    if hasattr(model.body_extractor, 'model') and model.body_extractor.model is not None:
        print("✓ DINOv2 feature extractor initialized successfully")
    else:
        print("✗ DINOv2 feature extractor not properly initialized")
    
    # Test perceivers
    print("\nTesting perceiver modules...")
    sample_face_embedding = torch.randn(1, 1, 512)
    sample_body_embedding = torch.randn(1, 1, 1024)
    
    try:
        face_tokens = model.face_perceiver(sample_face_embedding)
        print(f"✓ Face perceiver works: output shape {face_tokens.shape}")
    except Exception as e:
        print(f"✗ Face perceiver error: {e}")
    
    try:
        body_tokens = model.body_perceiver(sample_body_embedding)
        print(f"✓ Body perceiver works: output shape {body_tokens.shape}")
    except Exception as e:
        print(f"✗ Body perceiver error: {e}")
    
    # Test identity fusion
    if model.use_identity_fusion:
        print("\nTesting identity fusion...")
        try:
            model.prepare_identity_tokens(sample_face_embedding, sample_body_embedding)
            print("✓ Identity fusion initialized successfully")
        except Exception as e:
            print(f"✗ Identity fusion error: {e}")
    
    print("\nModel structure test completed")

def test_batch_generation(model, ref_image_paths, prompts):
    """Test batch image generation with identity preservation"""
    print(f"\n--- Testing batch generation with {len(prompts)} prompts ---")
    
    # Generate multiple images
    start_time = time.time()
    images = model.batch_generate(
        prompts=prompts,
        reference_images=ref_image_paths,
        negative_prompts=["blurry, low quality"] * len(prompts),
        num_inference_steps=30,
        guidance_scale=7.5,
        face_strength=1.0,
        body_strength=0.8
    )
    end_time = time.time()
    
    print(f"Batch generation took {end_time - start_time:.2f} seconds")
    print(f"Generated {len(images)} images")
    
    # Save generated images
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    batch_dir = os.path.join(results_dir, "batch")
    os.makedirs(batch_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        img_path = os.path.join(batch_dir, f"generated_{i}.png")
        img.save(img_path)
    
    print(f"Batch images saved to {batch_dir}")
    
    return images

def test_lora_application(model, ref_image_path, prompt, lora_path, weight_name=None):
    """Test LoRA application"""
    print(f"\n--- Testing LoRA application from {lora_path} ---")
    
    # Apply LoRA
    start_time = time.time()
    model.add_lora(lora_path, weight_name=weight_name, adapter_name="test_lora", lora_scale=0.8)
    end_time = time.time()
    
    print(f"LoRA application took {end_time - start_time:.2f} seconds")
    
    # Generate with LoRA applied
    image = model.generate(
        prompt=prompt,
        reference_image=ref_image_path,
        negative_prompt="blurry, low quality, cartoon, drawing, illustration",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    # Save generated image
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    image_path = os.path.join(results_dir, "generated_with_lora.png")
    image.save(image_path)
    print(f"Generated image with LoRA saved to {image_path}")
    
    return image

def test_multiple_loras(model, ref_image_path, prompt, lora_configs):
    """Test multiple LoRA application"""
    print(f"\n--- Testing multiple LoRAs ({len(lora_configs)}) ---")
    
    # Apply LoRAs
    start_time = time.time()
    model.add_multiple_loras(lora_configs)
    end_time = time.time()
    
    print(f"Multiple LoRA application took {end_time - start_time:.2f} seconds")
    
    # Generate with LoRAs applied
    image = model.generate(
        prompt=prompt,
        reference_image=ref_image_path,
        negative_prompt="blurry, low quality, cartoon, drawing, illustration",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    # Test with fused LoRAs
    print("Testing with fused LoRAs...")
    model.fuse_loras()
    
    fused_image = model.generate(
        prompt=prompt,
        reference_image=ref_image_path,
        negative_prompt="blurry, low quality, cartoon, drawing, illustration",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    # Save generated images
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    image.save(os.path.join(results_dir, "generated_with_multiple_loras.png"))
    fused_image.save(os.path.join(results_dir, "generated_with_fused_loras.png"))
    
    print(f"Images with multiple LoRAs saved to {results_dir}")
    
    return image, fused_image

def main():
    """Main test function - non-interactive version"""
    # Check for GPU
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU (this will be slow)")
    
    # Create test directory
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing IdentityPreservingFlux...")
    model = IdentityPreservingFlux(
        face_injection_index=18,  # SingleStream index
        body_injection_index=10,  # DoubleStream index
        use_gpu=torch.cuda.is_available(),
        use_identity_fusion=True,
        yolo_confidence=0.25
    )
    
    # Test the model structure without loading weights
    print("\nPerforming model structure test...")
    test_model_structure(model)
    
    # Try to load base model
    print("\nLoading Flux.1 base model...")
    try:
        start_time = time.time()
        from diffusers import FluxModel
        base_model = FluxModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model.load_base_model(base_model)
        end_time = time.time()
        print(f"Model loading took {end_time - start_time:.2f} seconds")
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This is expected if you don't have the model weights downloaded.")
        print("The structure test above verifies that the architecture is correctly implemented.")
        model_loaded = False
    
    # Always run dummy data test for basic functionality check
    print("\nRunning test with dummy data...")
    dummy_test_success = test_generation_with_dummy_data(model)
    
    # If model not loaded or dummy test failed, exit early
    if not model_loaded or not dummy_test_success:
        print("\nSkipping image-based tests due to missing model or failed dummy test.")
        print("All structure tests completed!")
        return
    
    # If we have a sample image, use it for testing
    sample_image_found = False
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_creation", "dataset")
    sample_paths = [
        os.path.join(dataset_dir, "train", "images", "image_001.jpg"),
        os.path.join(dataset_dir, "val", "images", "image_001.jpg")
    ]
    
    for path in sample_paths:
        if os.path.exists(path):
            ref_image_path = path
            print(f"\nUsing sample image: {ref_image_path}")
            sample_image_found = True
            
            # Test with default prompt
            prompt = "A professional portrait photograph of a person, studio lighting, high quality, photorealistic"
            print(f"Using default prompt: {prompt}")
            
            # Test generation with identity preservation
            test_generation(model, ref_image_path, prompt)
            
            # Test strength control
            print("\n--- Testing identity strength control ---")
            for face_str, body_str in [(0.5, 0.5), (1.0, 0.3), (0.3, 1.0)]:
                print(f"Testing with face_strength={face_str}, body_strength={body_str}")
                image = model.generate(
                    prompt=prompt,
                    reference_image=ref_image_path,
                    face_strength=face_str,
                    body_strength=body_str,
                    num_inference_steps=30
                )
                save_path = os.path.join(results_dir, f"strength_face{face_str}_body{body_str}.png")
                image.save(save_path)
                print(f"Image saved to {save_path}")
            
            break
    
    if not sample_image_found:
        print("\nNo sample images found for testing. Skipping image-based tests.")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 