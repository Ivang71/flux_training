import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from pathlib import Path

from architecture import IdentityPreservingFlux

device = "cuda"

def test_identity_extraction(model, ref_image_path):
    print(f"\n--- Testing identity extraction from {ref_image_path} ---")
    
    ref_img = Image.open(ref_image_path).convert("RGB")
    
    start_time = time.time()
    face_embeddings, body_embeddings, face_metadata, body_boxes = model.extract_identity(ref_img)
    end_time = time.time()
    
    print(f"Identity extraction took {end_time - start_time:.2f} seconds")
    print(f"Found {face_embeddings.shape[1]} faces")
    print(f"Found {body_embeddings.shape[1]} person bodies")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(np.array(ref_img))
    
    for face in face_metadata:
        x1, y1, x2, y2 = [int(coord) for coord in face["bbox"]]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    for box in body_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
    
    ax.set_title(f"Detections: {face_embeddings.shape[1]} faces, {body_embeddings.shape[1]} bodies")
    ax.axis('off')
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "detections.png"))
    plt.close(fig)
    
    print(f"Detection visualization saved to {os.path.join(results_dir, 'detections.png')}")
    
    return face_embeddings, body_embeddings

def test_generation(model, ref_image_path, prompt, save_visualization=True):
    print(f"\n--- Testing generation with prompt: '{prompt}' ---")
    
    face_embeddings, body_embeddings = test_identity_extraction(model, ref_image_path)
    
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
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results")
    os.makedirs(results_dir, exist_ok=True)
    image_path = os.path.join(results_dir, "generated.png")
    image.save(image_path)
    print(f"Generated image saved to {image_path}")
    
    if save_visualization:
        viz_path = os.path.join(results_dir, "attention_visualization.png")
        model.save_visualization(viz_path)
        print(f"Attention visualization saved to {viz_path}")
    
    return image

def test_generation_with_dummy_data(model):
    print("\n--- Testing with dummy identity data ---")
    
    cuda_device = device if device.startswith("cuda") else None
    device_to_use = cuda_device if torch.cuda.is_available() else "cpu"
    
    face_embeddings = torch.randn(1, 1, 512, device=device_to_use)
    body_embeddings = torch.randn(1, 1, 1024, device=device_to_use)
    
    print(f"Created dummy tensors on device: {device_to_use}")
    
    try:
        try:
            model.face_perceiver = model.face_perceiver.to(device_to_use)
            face_tokens = model.face_perceiver(face_embeddings)
            print(f"Face perceiver output shape: {face_tokens.shape}, device: {face_tokens.device}")
        except Exception as e:
            print(f"Error in face perceiver: {e}")
            return False
        
        try:
            model.body_perceiver = model.body_perceiver.to(device_to_use)
            body_tokens = model.body_perceiver(body_embeddings)
            print(f"Body perceiver output shape: {body_tokens.shape}, device: {body_tokens.device}")
        except Exception as e:
            print(f"Error in body perceiver: {e}")
            return False
        
        if model.use_identity_fusion:
            try:
                face_processed = model.face_perceiver(face_embeddings)
                body_processed = model.body_perceiver(body_embeddings)
                
                model.identity_fusion = model.identity_fusion.to(device_to_use)
                
                fused = model.identity_fusion(face_processed, body_processed)
                print(f"Identity fusion output shape: {fused.shape}, device: {fused.device}")
                
                expected_seq_len = face_processed.size(1) + body_processed.size(1)
                if fused.size(1) != expected_seq_len:
                    print(f"WARNING: Identity fusion output has unexpected sequence length: {fused.size(1)} vs expected {expected_seq_len}")
            except Exception as e:
                print(f"Error in identity fusion module: {e}")
                print("This is the component with the dimension mismatch issue.")
                return False
        
        try:
            model.face_injection = model.face_injection.to(device_to_use)
            model.body_injection = model.body_injection.to(device_to_use)
            
            model.prepare_identity_tokens(face_embeddings, body_embeddings)
            print("Identity tokens prepared successfully")
            
            if hasattr(model, "current_face_tokens"):
                print(f"Stored face tokens shape: {model.current_face_tokens.shape}, device: {model.current_face_tokens.device}")
            if hasattr(model, "current_body_tokens"):
                print(f"Stored body tokens shape: {model.current_body_tokens.shape}, device: {model.current_body_tokens.device}")
        except Exception as e:
            print(f"Error in prepare_identity_tokens: {e}")
            return False
            
        print("\nSkipping full inference test - focusing on component testing only")
        print("Model structure testing completed successfully!")
        return True
    except Exception as e:
        print(f"Error in dummy data test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_structure(model):
    print("\n--- Testing model structure ---")
    
    print(f"Face injection index: {model.face_injection_index}")
    print(f"Body injection index: {model.body_injection_index}")
    
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
    
    if model.use_identity_fusion:
        print("\nTesting identity fusion...")
        try:
            model.prepare_identity_tokens(sample_face_embedding, sample_body_embedding)
            print("✓ Identity fusion initialized successfully")
        except Exception as e:
            print(f"✗ Identity fusion error: {e}")
    
    print("\nModel structure test completed")

def test_batch_generation(model, ref_image_paths, prompts):
    print(f"\n--- Testing batch generation with {len(prompts)} prompts ---")
    
    start_time = time.time()
    images = model.batch_generate(
        prompts=prompts,
        reference_images=ref_image_paths,
        negative_prompts=["blurry, low quality"] * len(prompts),
        num_inference_steps=30,
        guidance_scale=7.5,
        face_strength=1.0,
        body_strength=1.0
    )
    end_time = time.time()
    
    print(f"Batch generation took {end_time - start_time:.2f} seconds for {len(prompts)} images")
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results", "batch")
    os.makedirs(results_dir, exist_ok=True)
    
    for i, image in enumerate(images):
        image_path = os.path.join(results_dir, f"generated_{i+1}.png")
        image.save(image_path)
    
    print(f"Batch generated images saved to {results_dir}")
    
    return images

def test_lora_application(model, ref_image_path, prompt, lora_path, weight_name=None):
    print(f"\n--- Testing LoRA application with {lora_path} ---")
    
    start_time = time.time()
    model.add_lora(
        lora_path=lora_path,
        weight_name=weight_name,
        adapter_name="test_lora",
        lora_scale=0.8
    )
    end_time = time.time()
    
    print(f"LoRA application took {end_time - start_time:.2f} seconds")
    
    image = model.generate(
        prompt=prompt,
        reference_image=ref_image_path,
        negative_prompt="blurry, low quality, cartoon, drawing, illustration",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results", "lora")
    os.makedirs(results_dir, exist_ok=True)
    image_path = os.path.join(results_dir, "lora_generated.png")
    image.save(image_path)
    
    print(f"LoRA generated image saved to {image_path}")
    
    return image

def test_multiple_loras(model, ref_image_path, prompt, lora_configs):
    print(f"\n--- Testing multiple LoRAs application with {len(lora_configs)} LoRAs ---")
    
    start_time = time.time()
    model.add_multiple_loras(lora_configs)
    end_time = time.time()
    
    print(f"Multiple LoRAs application took {end_time - start_time:.2f} seconds")
    
    image = model.generate(
        prompt=prompt,
        reference_image=ref_image_path,
        negative_prompt="blurry, low quality, cartoon, drawing, illustration",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_results", "lora")
    os.makedirs(results_dir, exist_ok=True)
    image_path = os.path.join(results_dir, "multi_lora_generated.png")
    image.save(image_path)
    
    print(f"Multiple LoRAs generated image saved to {image_path}")
    
    start_time = time.time()
    model.fuse_loras()
    end_time = time.time()
    
    print(f"LoRA fusion took {end_time - start_time:.2f} seconds")
    
    fused_image = model.generate(
        prompt=prompt,
        reference_image=ref_image_path,
        negative_prompt="blurry, low quality, cartoon, drawing, illustration",
        num_inference_steps=30,
        guidance_scale=7.5
    )
    
    fused_image_path = os.path.join(results_dir, "fused_lora_generated.png")
    fused_image.save(fused_image_path)
    
    print(f"Fused LoRAs generated image saved to {fused_image_path}")
    
    return image, fused_image

def main():
    print("Initializing IdentityPreservingFlux model...")
    model = IdentityPreservingFlux(
        use_gpu=torch.cuda.is_available(),
        yolo_confidence=0.25
    )
    model = model.to(device)
    
    print("Loading base model...")
    model.load_base_model()
    
    test_model_structure(model)
    
    test_passed = test_generation_with_dummy_data(model)
    if not test_passed:
        print("Dummy data test failed. Exiting to avoid further errors.")
        return
    
    if len(sys.argv) > 1:
        ref_image_path = sys.argv[1]
        
        if os.path.exists(ref_image_path):
            print(f"Using provided reference image: {ref_image_path}")
            
            if len(sys.argv) > 2:
                prompt = sys.argv[2]
            else:
                prompt = "A high quality photo of a person with the same identity in a different setting"
                
            test_generation(model, ref_image_path, prompt)
            
            if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
                lora_path = sys.argv[3]
                try:
                    test_lora_application(model, ref_image_path, prompt, lora_path)
                except Exception as e:
                    print(f"Error testing LoRA application: {e}")
                    import traceback
                    traceback.print_exc()
                
            if len(sys.argv) > 4:
                ref_image_paths = [ref_image_path]
                prompts = [prompt]
                for i in range(4, min(len(sys.argv), 7)):
                    if os.path.exists(sys.argv[i]):
                        ref_image_paths.append(sys.argv[i])
                        prompts.append(f"A photo of the person in an outdoor setting, style {i-3}")
                
                if len(ref_image_paths) > 1:
                    try:
                        test_batch_generation(model, ref_image_paths, prompts)
                    except Exception as e:
                        print(f"Error testing batch generation: {e}")
                        import traceback
                        traceback.print_exc()
        else:
            print(f"Reference image not found: {ref_image_path}")
            print("Skipping generation tests that require a real image")
    else:
        print("No reference image provided. Skipping generation tests.")
        print("Usage: python test.py <reference_image> [prompt] [lora_path] [additional_ref_images]")
    
    print("\nTesting completed!")

if __name__ == "__main__":
    main() 