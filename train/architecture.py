import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from insightface.app import FaceAnalysis
import numpy as np
import os
from typing import Optional, List, Tuple, Dict, Any, Union
import matplotlib.pyplot as plt
from PIL import Image
import time

from diffusers import FluxPipeline

class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler module that transforms variable-length input embeddings 
    into a fixed number of latent tokens.
    """
    def __init__(self, input_dim: int, latent_dim: int, num_latents: int, num_layers: int = 4, num_heads: int = 8, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=num_heads,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D_input]
        batch_size = x.shape[0]
        
        # Project input to latent dimension if needed
        if x.shape[-1] != self.latents.shape[-1]:
            x = self.input_proj(x)
        
        # Initialize latents for each batch item
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, D_latent]
        
        # Process through cross-attention layers
        for layer, norm in zip(self.layers, self.norm_layers):
            # Cross-attention: latents attend to input
            attn_output, _ = layer(
                query=latents,
                key=x,
                value=x
            )
            latents = norm(latents + attn_output)
            
        return latents


class IdentityInjectionLayer(nn.Module):
    """
    Cross-attention layer that injects identity information into UNet feature maps.
    """
    def __init__(self, feature_dim: int, identity_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.identity_dim = identity_dim
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.identity_proj = nn.Linear(identity_dim, feature_dim)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.GELU(),
            nn.Linear(4 * feature_dim, feature_dim)
        )
        
        # For storing attention maps for visualization
        self.last_attention_map = None
    
    def forward(self, hidden_states: torch.Tensor, identity_tokens: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        # Store original shape for later reshape
        original_shape = hidden_states.shape
        original_dtype = hidden_states.dtype
        
        # Print shape information for debugging
        print(f"Hidden states shape: {hidden_states.shape}, Identity tokens shape: {identity_tokens.shape}")
        print(f"Hidden states dtype: {hidden_states.dtype}, Identity tokens dtype: {identity_tokens.dtype}")
        
        # Adapt the hidden states to the expected format [B, seq_len, dim]
        if len(original_shape) == 4:  # [B, C, H, W]
            B, C, H, W = original_shape
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(B, H*W, C)
        elif len(original_shape) == 3 and original_shape[2] != self.feature_dim:
            # If the last dimension doesn't match feature_dim, reshape is needed
            B, seq_len, C = original_shape
            # Use reshape or projection based on dimensions
            if C % self.feature_dim == 0:
                # Reshape to merge dimensions
                factor = C // self.feature_dim
                hidden_states = hidden_states.reshape(B, seq_len * factor, self.feature_dim)
            else:
                # Project to the correct dimension using a dynamic projection
                print(f"Creating dynamic projection from {C} to {self.feature_dim}")
                projection = nn.Linear(C, self.feature_dim).to(hidden_states.device)
                hidden_states = projection(hidden_states)
                
        # Convert to the same dtype if needed
        if hidden_states.dtype != identity_tokens.dtype:
            identity_tokens = identity_tokens.to(dtype=hidden_states.dtype)
        
        # Project identity tokens if dimensions don't match
        if identity_tokens.shape[-1] != self.feature_dim:
            identity_tokens = self.identity_proj(identity_tokens)
        
        # First run the layer norm on hidden states
        try:
            norm_hidden = self.norm1(hidden_states)
        except Exception as e:
            print(f"Error in norm1: {e}")
            print(f"Using custom normalization due to dimension mismatch")
            # Apply a custom normalization as fallback
            norm_hidden = F.layer_norm(
                hidden_states, 
                normalized_shape=[hidden_states.shape[-1]], 
                weight=None, 
                bias=None
            )
        
        # Attention block
        try:
            attn_output, attn_weights = self.attn(
                query=norm_hidden,
                key=identity_tokens,
                value=identity_tokens
            )
            
            # Store attention map for visualization
            self.last_attention_map = attn_weights
            
            # Apply identity strength control
            hidden_states = hidden_states + (attn_output * strength)
        except Exception as e:
            print(f"Error in attention: {e}")
            print("Skipping attention due to dimension mismatch")
            # Return original hidden states if attention fails
            return hidden_states
        
        # Feed-forward block
        try:
            norm_hidden = self.norm2(hidden_states)
            ff_output = self.ff(norm_hidden)
            hidden_states = hidden_states + ff_output
        except Exception as e:
            print(f"Error in feed-forward: {e}")
            print("Skipping feed-forward due to dimension mismatch")
        
        # Reshape back to the original dimensions if needed
        if len(original_shape) == 4:  # [B, C, H, W]
            B, HW, C = hidden_states.shape
            H = W = int(np.sqrt(HW))
            hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif len(original_shape) == 3 and original_shape[2] != self.feature_dim:
            # Try to reshape back to original dimensions
            B, seq_len_new, C_new = hidden_states.shape
            if original_shape[2] % C_new == 0:
                factor = original_shape[2] // C_new
                hidden_states = hidden_states.reshape(B, original_shape[1], original_shape[2])
            
        # Convert back to original dtype if needed
        if hidden_states.dtype != original_dtype:
            hidden_states = hidden_states.to(dtype=original_dtype)
            
        return hidden_states


class IdentityFusionModule(nn.Module):
    """
    Module to fuse face and body identity features before injection.
    """
    def __init__(self, face_dim=1024, body_dim=1024, hidden_dim=1024):
        super().__init__()
        self.face_dim = face_dim
        self.body_dim = body_dim
        self.hidden_dim = hidden_dim
        
        # Self-attention for feature fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
            
    def forward(self, face_tokens, body_tokens):
        """
        Fuse face and body tokens using self-attention.
        
        Args:
            face_tokens: Tensor of shape [batch_size, num_face_tokens, dim]
            body_tokens: Tensor of shape [batch_size, num_body_tokens, dim]
            
        Returns:
            Tensor of shape [batch_size, num_face_tokens + num_body_tokens, dim]
        """
        # Normalize inputs - shape [B, num_tokens, feature_dim]
        face_tokens = self.layer_norm(face_tokens)
        body_tokens = self.layer_norm(body_tokens)
        
        # Concatenate along sequence dimension
        # Shape: [B, num_face_tokens + num_body_tokens, dim]
        combined = torch.cat([face_tokens, body_tokens], dim=1)
        
        # Self-attention with same tensor for query, key, value
        # Input shape: [B, seq_len, hidden_dim]
        # Output shape: [B, seq_len, hidden_dim]
        attn_output, _ = self.attention(combined, combined, combined)
        
        # Add residual connection
        output = combined + attn_output
        
        return output


class FaceIdentityExtractor(nn.Module):
    """
    Module to extract face identity embeddings using InsightFace.
    """
    def __init__(self, use_gpu: bool = True):
        super().__init__()
        self.face_analyzer = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider'],
            name="buffalo_l"  # Using the higher quality model
        )
        self.face_analyzer.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    
    def forward(self, img: np.ndarray) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Extract face embedding from an image (only the primary/largest face).
        
        Args:
            img: np.ndarray of shape [H, W, C] in RGB format
        
        Returns:
            face_embedding: torch.Tensor of shape [1, 1, 512] (single face)
            face_metadata: List with single face information dictionary or empty list
        """
        # Get the device from the model parameters
        device = next(self.parameters(), torch.tensor(0)).device
        
        faces = self.face_analyzer.get(img)
        
        if not faces:
            # Return empty embedding if no face detected
            return torch.zeros(1, 1, 512, device=device), []
        
        # Sort faces by area (largest first)
        faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        
        # Get embedding only for the largest face
        primary_face = faces[0]
        face_embedding = torch.tensor(primary_face.embedding, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
        
        # Prepare face metadata for the primary face
        metadata = [{
            "bbox": primary_face.bbox,
            "kps": primary_face.kps,
            "det_score": primary_face.det_score,
            "landmark_3d_68": getattr(primary_face, "landmark_3d_68", None),
            "gender": getattr(primary_face, "gender", None),
            "age": getattr(primary_face, "age", None)
        }]
        
        return face_embedding, metadata


class YOLOBodyDetector(nn.Module):
    """
    Module to detect people using YOLOv8
    """
    def __init__(self, model_size: str = "m", confidence: float = 0.25, cache_dir: Optional[str] = None):
        """
        Initialize the YOLOv8 detector.
        
        Args:
            model_size: Size of YOLOv8 model ('n', 's', 'm', 'l', 'x')
            confidence: Confidence threshold for detections
            cache_dir: Directory to cache models
        """
        super().__init__()
        self.model = None
        self.model_size = model_size
        self.confidence = confidence
        # Use a local cache directory in the current directory
        self.cache_dir = cache_dir or os.path.join(".", "cache", "identity_preserving")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_model()
    
    def _load_model(self):
        """Lazy load YOLOv8 model to avoid import errors if not needed"""
        try:
            from ultralytics import YOLO
            model_path = f"yolov8{self.model_size}.pt"
            self.model = YOLO(model_path)
        except ImportError:
            print("YOLOv8 not installed. Please install with: pip install ultralytics")
            self.model = None
    
    def detect_person(self, img: np.ndarray) -> List[torch.Tensor]:
        """
        Detect person bounding boxes using YOLOv8.
        
        Args:
            img: np.ndarray of shape [H, W, C] in RGB format
            
        Returns:
            List of tensors containing [x1, y1, x2, y2, score]
        """
        if self.model is None:
            self._load_model()
            if self.model is None:
                return []
        
        # Run inference
        results = self.model(img, conf=self.confidence, verbose=False)
        
        # Extract person detections (class 0 in COCO)
        person_boxes = []
        
        for result in results:
            # Filter for person class (0)
            boxes = result.boxes
            for i, cls in enumerate(boxes.cls):
                if int(cls) == 0:  # Person class
                    box = boxes.xyxy[i].tolist()  # x1, y1, x2, y2
                    score = float(boxes.conf[i])
                    person_boxes.append(torch.tensor([*box, score]))
        
        return person_boxes


class BodyIdentityExtractor(nn.Module):
    """
    Module to extract body features using YOLOv8 for detection and DINOv2 for features.
    """
    def __init__(self, cache_dir: Optional[str] = None, use_gpu: bool = True, yolo_confidence: float = 0.25):
        super().__init__()
        # Use a local cache directory in the current directory
        self.cache_dir = cache_dir or os.path.join(".", "cache", "identity_preserving")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.detector = YOLOBodyDetector(confidence=yolo_confidence, cache_dir=self.cache_dir)
        
        # Load DINOv2 for feature extraction
        self.processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large")
        
        # Move to GPU if available and requested
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            
        # Flag for caching
        self.use_cached = False
        self.cached_embeddings = {}
    
    def enable_caching(self, enabled: bool = True):
        """Enable or disable caching of embeddings"""
        self.use_cached = enabled
        if not enabled:
            self.cached_embeddings = {}
    
    def extract_features(self, img: np.ndarray, box: torch.Tensor) -> torch.Tensor:
        """
        Extract body features using DINOv2.
        
        Args:
            img: np.ndarray of shape [H, W, C] in RGB format
            box: torch.Tensor containing [x1, y1, x2, y2, score]
            
        Returns:
            features: torch.Tensor of shape [1, 1024]
        """
        # Get the device from the model parameters
        device = self.model.device
        
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [max(0, int(coord)) for coord in box[:4]]
        
        # Crop the image
        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0:  # If the crop is empty
            return torch.zeros(1, 1024, device=device)
        
        # Check cache if enabled
        if self.use_cached:
            # Simple hash of crop content
            crop_hash = hash(cropped_img.tobytes())
            if crop_hash in self.cached_embeddings:
                # Ensure cached result is on the correct device
                return self.cached_embeddings[crop_hash].to(device)
        
        # Preprocess the image
        inputs = self.processor(images=cropped_img, return_tensors="pt")
        
        # Move to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get CLS token features
            features = outputs.last_hidden_state[:, 0]  # [1, 1024]
        
        # Cache the result if enabled
        if self.use_cached:
            self.cached_embeddings[crop_hash] = features.detach().clone()
        
        return features
    
    def forward(self, img: np.ndarray) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Extract body embedding from an image (only the primary/largest person).
        
        Args:
            img: np.ndarray of shape [H, W, C] in RGB format
            
        Returns:
            body_embedding: torch.Tensor of shape [1, 1, 1024] (single person)
            body_box: List with single tensor containing [x1, y1, x2, y2, score] or empty list
        """
        # Get the device from the model parameters
        device = self.model.device
        
        start_time = time.time()
        
        # Detect persons
        boxes = self.detector.detect_person(img)
        
        detect_time = time.time()
        print(f"Person detection took: {detect_time - start_time:.2f}s")
        
        if not boxes:
            # Return empty embedding if no person detected
            return torch.zeros(1, 1, 1024, device=device), []
        
        # Sort boxes by area (largest first) if there are multiple detections
        if len(boxes) > 1:
            boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        
        # Extract features only for the largest person
        primary_box = boxes[0]
        body_embedding = self.extract_features(img, primary_box).unsqueeze(0)  # [1, 1, 1024]
        
        feature_time = time.time()
        print(f"Feature extraction took: {feature_time - detect_time:.2f}s")
        
        return body_embedding, [primary_box]


class IdentityPreservingFlux(nn.Module):
    """
    Identity-preserving system for Flux.1.
    
    Based on the actual Flux.1 architecture as shown in the diagram.
    """
    def __init__(
        self,
        face_embedding_dim: int = 512,
        body_embedding_dim: int = 1024,
        num_face_latents: int = 16,
        num_body_latents: int = 16,
        num_fused_latents: int = 32,
        face_injection_index: int = 18,  # Near the end of the SingleStream blocks
        body_injection_index: int = 10,  # Earlier in the DoubleStream blocks
        use_gpu: bool = True,
        cache_dir: Optional[str] = None,
        use_identity_fusion: bool = True,
        yolo_confidence: float = 0.25
    ):
        super().__init__()
        
        # Initialize identity extractors
        self.face_extractor = FaceIdentityExtractor(use_gpu=use_gpu)
        self.body_extractor = BodyIdentityExtractor(
            cache_dir=cache_dir, 
            use_gpu=use_gpu,
            yolo_confidence=yolo_confidence
        )
        
        # Set up dtype and device for model loading
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device_map = "auto" if torch.cuda.is_available() else None
        
        # Perceiver Resampler modules for identity embeddings
        self.face_perceiver = PerceiverResampler(
            input_dim=face_embedding_dim,
            latent_dim=1024,  # Using 1024 as the standard dimension for Flux.1
            num_latents=num_face_latents
        )
        
        self.body_perceiver = PerceiverResampler(
            input_dim=body_embedding_dim,
            latent_dim=1024,
            num_latents=num_body_latents
        )
        
        # Identity fusion module (new)
        self.use_identity_fusion = use_identity_fusion
        if use_identity_fusion:
            self.identity_fusion = IdentityFusionModule(face_dim=1024, body_dim=1024)
            # Additional perceiver for fused tokens
            self.fused_perceiver = PerceiverResampler(
                input_dim=1024,
                latent_dim=1024,
                num_latents=num_fused_latents
            )
        
        # Identity injection layers
        self.face_injection = IdentityInjectionLayer(
            feature_dim=1024,
            identity_dim=1024
        )
        
        self.body_injection = IdentityInjectionLayer(
            feature_dim=1024,
            identity_dim=1024
        )
        
        # Indices for where to inject identity information
        self.face_injection_index = face_injection_index
        self.body_injection_index = body_injection_index
        
        # Will store base model reference when loaded
        self.base_model = None
        
        # Initialize hooks dict to properly handle unregistering
        self.hooks = {}
        
        # Identity strength control
        self.face_strength = 1.0
        self.body_strength = 1.0
        
        # Training mode flag (to avoid collision with YOLO train method)
        self.training_mode = True
    
    def set_training_mode(self, mode=True):
        """
        Set the module in training or evaluation mode.
        This method overrides nn.Module.train() to avoid conflicts with YOLO.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        
        Returns:
            self
        """
        self.training_mode = mode
        # Apply the mode to all submodules except those with their own train method
        for module in self.children():
            if not hasattr(module, 'train') or not callable(module.train) or isinstance(module, nn.Module):
                if hasattr(module, 'training'):
                    module.training = mode
        
        return self
        
    def train(self, mode=True):
        """
        Override the train method to use our custom set_training_mode
        to avoid collisions with YOLO's train method.
        """
        return self.set_training_mode(mode)
    
    def eval(self):
        """
        Override the eval method to use our custom set_training_mode.
        """
        return self.set_training_mode(False)
    
    def set_identity_strength(self, face_strength: float = 1.0, body_strength: float = 1.0):
        """Set the strength of identity preservation (0.0-1.0)"""
        self.face_strength = max(0.0, min(1.0, face_strength))
        self.body_strength = max(0.0, min(1.0, body_strength))
    
    def _get_flux_module_by_index(self, idx: int) -> nn.Module:
        """
        Get the appropriate module from Flux.1 by index.
        
        Based on the architecture diagram:
        - 0-18: 19 DoubleStream blocks (N = 19)
        - 19-56: 38 SingleStream blocks (M = 38)
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        # Access blocks directly from the model's layers for compatibility
        # with different versions of the Flux model
        try:
            # For newer versions of Flux using transformer blocks
            if hasattr(self.base_model, 'transformer_blocks'):
                return self.base_model.transformer_blocks[idx]
            # For versions with separate double_stream and single_stream blocks
            elif hasattr(self.base_model, 'double_stream_blocks') and idx < 19:
                return self.base_model.double_stream_blocks[idx]
            elif hasattr(self.base_model, 'single_stream_blocks') and idx >= 19 and idx < 57:
                return self.base_model.single_stream_blocks[idx - 19]
            # For models with a flat list of blocks
            elif hasattr(self.base_model, 'blocks'):
                return self.base_model.blocks[idx]
            # For models with layers attribute
            elif hasattr(self.base_model, 'layers'):
                return self.base_model.layers[idx]
            else:
                print(f"WARNING: Could not find appropriate blocks in the model structure.")
                print(f"Available attributes: {dir(self.base_model)}")
                # Return the model itself as fallback
                return self.base_model
        except (IndexError, AttributeError) as e:
            print(f"Error accessing model layer at index {idx}: {e}")
            print(f"Model structure may have changed. Available attributes: {dir(self.base_model)}")
            return self.base_model
    
    def _register_hooks(self):
        """Register forward hooks in the base model to inject identity information"""
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        # Unregister existing hooks if any
        self._unregister_hooks()
        
        # Try to retrieve the modules for face and body injection
        try:
            face_module = self._get_flux_module_by_index(self.face_injection_index)
            body_module = self._get_flux_module_by_index(self.body_injection_index)
            
            print(f"Successfully located face injection module: {type(face_module).__name__}")
            print(f"Successfully located body injection module: {type(body_module).__name__}")
        except Exception as e:
            print(f"Error finding modules for hooks: {e}")
            print("Attempting alternate hook registration...")
            
            # Fallback: Register hooks on the entire model
            face_module = self.base_model
            body_module = self.base_model
            print("Using the entire model for hooks (fallback)")
            
        # Define face hook
        def face_hook(module, input_tensors, output):
            try:
                # Get the tensors in the format needed
                hidden_states = output
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                
                # Skip if we don't have face tokens
                if not hasattr(self, 'current_face_tokens'):
                    return output
                    
                # Apply identity injection with error handling
                try:
                    hidden_states = self.face_injection(
                        hidden_states, 
                        self.current_face_tokens,
                        strength=self.face_strength
                    )
                    
                    # Reshape result to match output format
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states
                except Exception as e:
                    print(f"Error in face injection: {e}")
                    # On error, return original output
                    return output
            except Exception as e:
                print(f"Error in face hook: {e}")
                # On any error, return original output
                return output
            
        # Define body hook
        def body_hook(module, input_tensors, output):
            try:
                # Get the tensors in the format needed
                hidden_states = output
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                
                # Skip if we don't have body tokens
                if not hasattr(self, 'current_body_tokens'):
                    return output
                
                # Apply identity injection with error handling
                try:
                    hidden_states = self.body_injection(
                        hidden_states, 
                        self.current_body_tokens,
                        strength=self.body_strength
                    )
                    
                    # Reshape result to match output format
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states
                except Exception as e:
                    print(f"Error in body injection: {e}")
                    # On error, return original output
                    return output
            except Exception as e:
                print(f"Error in body hook: {e}")
                # On any error, return original output
                return output
        
        # Register the hooks
        try:
            self.hooks['face'] = face_module.register_forward_hook(face_hook)
            self.hooks['body'] = body_module.register_forward_hook(body_hook)
            print("Successfully registered identity injection hooks")
        except Exception as e:
            print(f"Failed to register hooks: {e}")
            print("Identity injection may not work properly")
    
    def _unregister_hooks(self):
        """Unregister all hooks"""
        for hook_name, hook in self.hooks.items():
            hook.remove()
        self.hooks = {}
    
    def _scan_model_structure(self):
        """Scan the model structure to identify appropriate injection points"""
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
            
        print("\nScanning Flux model structure...")
        
        # Check for specific Flux model architecture components
        if hasattr(self.base_model, 'transformer_blocks'):
            blocks = self.base_model.transformer_blocks
            print(f"Found transformer_blocks: {len(blocks)} blocks")
            # Use appropriate indices for different points in the model
            total_blocks = len(blocks)
            self.face_injection_index = int(total_blocks * 0.9)  # Face near the end (90%)
            self.body_injection_index = int(total_blocks * 0.5)  # Body in the middle (50%)
            print(f"Set face_injection_index={self.face_injection_index}, body_injection_index={self.body_injection_index}")
            return True
            
        elif hasattr(self.base_model, 'blocks'):
            blocks = self.base_model.blocks
            print(f"Found blocks: {len(blocks)} blocks")
            total_blocks = len(blocks)
            self.face_injection_index = int(total_blocks * 0.9)
            self.body_injection_index = int(total_blocks * 0.5)
            print(f"Set face_injection_index={self.face_injection_index}, body_injection_index={self.body_injection_index}")
            return True
            
        elif hasattr(self.base_model, 'layers'):
            layers = self.base_model.layers
            print(f"Found layers: {len(layers)} layers")
            total_layers = len(layers)
            self.face_injection_index = int(total_layers * 0.9)
            self.body_injection_index = int(total_layers * 0.5)
            print(f"Set face_injection_index={self.face_injection_index}, body_injection_index={self.body_injection_index}")
            return True
            
        # If we have separate streams (original architecture)
        found_structure = False
        if hasattr(self.base_model, 'double_stream_blocks'):
            double_blocks = self.base_model.double_stream_blocks
            print(f"Found double_stream_blocks: {len(double_blocks)} blocks")
            found_structure = True
            
        if hasattr(self.base_model, 'single_stream_blocks'):
            single_blocks = self.base_model.single_stream_blocks
            print(f"Found single_stream_blocks: {len(single_blocks)} blocks")
            found_structure = True
            
        if found_structure:
            print(f"Original architecture detected. Using face_injection_index={self.face_injection_index}, body_injection_index={self.body_injection_index}")
            return True
            
        # If we couldn't identify the structure
        print("WARNING: Could not identify model structure.")
        print(f"Model attributes: {dir(self.base_model)}")
        print("Using default injection indices, but hooks may not work properly.")
        return False
        
    def load_base_model(self, base_model=None):
        """
        Load the base Flux model or use provided model.
        
        Args:
            base_model: Optional pre-loaded FluxModel instance
        
        Returns:
            The loaded base model
        """
        if base_model is not None:
            self.base_model = base_model
            # Scan model structure to determine injection points
            self._scan_model_structure()
            # Register hooks based on the determined structure
            self._register_hooks()
            return self.base_model
            
        try:
            print(f"Loading Flux model with torch_dtype={self.torch_dtype}")
            
            # Use FluxPipeline for proper loading
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Extract the transformer model
            self.base_model = pipe.transformer
            # Scan model structure to determine injection points
            self._scan_model_structure()
            # Register hooks based on the determined structure
            self._register_hooks()
            
            # Store the pipeline for future use
            self.pipeline = pipe
            
            return self.base_model
        except ImportError as e:
            print(f"Could not import FluxPipeline from diffusers: {e}")
            print("Make sure you have the latest version installed.")
            print("Continuing with model structure tests only.")
            return None
        except Exception as e:
            print(f"Error loading base model: {e}")
            print("Continuing with model structure tests only.")
            return None
    
    def extract_identity(self, image):
        """
        Extract face and body identity from an image or batch of images.
        
        Args:
            image: Either a PIL Image, numpy array, torch.Tensor, or path to an image file.
                 For tensors, can handle a batch [B, C, H, W] or a single image [C, H, W].
            
        Returns:
            face_embedding: torch.Tensor of shape [B, 1, 512] for batch or [1, 1, 512] for single image
            body_embedding: torch.Tensor of shape [B, 1, 1024] for batch or [1, 1, 1024] for single image
            face_metadata: List of dictionaries with face information (one per batch item or single)
            body_box: List of tensors containing [x1, y1, x2, y2, score] (one per batch item or single)
        """
        # Get device from the model
        device = next(self.parameters()).device
        
        # Special handling for tensor batch processing
        if isinstance(image, torch.Tensor) and image.dim() == 4 and image.shape[0] > 1:
            batch_size = image.shape[0]
            
            # Initialize outputs
            all_face_embeddings = []
            all_body_embeddings = []
            all_face_metadata = []
            all_body_boxes = []
            
            # Process each image in the batch
            for i in range(batch_size):
                single_img = image[i:i+1]  # Keep batch dimension as [1, C, H, W]
                
                # Convert tensor to numpy array for processing
                single_img_np = single_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Handle normalization
                if single_img_np.min() < 0 or single_img_np.max() > 1:
                    # Denormalize if image was normalized with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    mean = np.array([0.5, 0.5, 0.5])
                    std = np.array([0.5, 0.5, 0.5])
                    single_img_np = std * single_img_np + mean
                
                # Scale to [0, 255] and convert to uint8
                single_img_np = np.clip(single_img_np * 255, 0, 255).astype(np.uint8)
                
                # Process the numpy array
                face_emb, body_emb, face_meta, body_box = self._process_identity_from_numpy(single_img_np)
                
                # Move to same device as model
                face_emb = face_emb.to(device)
                body_emb = body_emb.to(device)
                
                all_face_embeddings.append(face_emb)
                all_body_embeddings.append(body_emb)
                all_face_metadata.append(face_meta)
                all_body_boxes.append(body_box)
                
            # Combine results - ensure all tensors are on the same device
            face_embedding = torch.cat(all_face_embeddings, dim=0).to(device)
            body_embedding = torch.cat(all_body_embeddings, dim=0).to(device)
            
            return face_embedding, body_embedding, all_face_metadata, all_body_boxes
        
        # Single image processing (including single tensor with batch dim = 1)
        # Convert image to numpy array if needed
        if isinstance(image, str):
            from PIL import Image
            image = np.array(Image.open(image).convert('RGB'))
        elif hasattr(image, 'mode'):  # PIL Image
            image = np.array(image.convert('RGB'))
        elif isinstance(image, torch.Tensor):  # PyTorch tensor
            # Handle single image with batch dimension
            if image.dim() == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
                
            # Convert tensor to numpy array
            image = image.permute(1, 2, 0).cpu().numpy()
            
            # Handle normalization
            if image.min() < 0 or image.max() > 1:
                # Denormalize if image was normalized with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                image = std * image + mean
                
            # Scale to [0, 255] and convert to uint8
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Process and move results to the correct device
        face_embedding, body_embedding, face_metadata, body_box = self._process_identity_from_numpy(image)
        face_embedding = face_embedding.to(device)
        body_embedding = body_embedding.to(device) 
        
        return face_embedding, body_embedding, face_metadata, body_box
    
    def _process_identity_from_numpy(self, img_np):
        """
        Helper method to process numpy array image for identity extraction.
        
        Args:
            img_np: Numpy array of shape [H, W, C] in RGB format
            
        Returns:
            face_embedding: torch.Tensor of shape [1, 1, 512]
            body_embedding: torch.Tensor of shape [1, 1, 1024]
            face_metadata: Dict with face information or {}
            body_box: Tensor containing [x1, y1, x2, y2, score] or None
        """
        # Get device from the model
        device = next(self.parameters()).device
        
        # Extract face embeddings
        try:
            face_embeddings, face_metadata = self.face_extractor(img_np)
            # Move embeddings to the correct device
            face_embeddings = face_embeddings.to(device)
        except Exception as e:
            print(f"Error in face extraction: {e}")
            # Return zero tensor if face extraction fails
            face_embeddings = torch.zeros(1, 1, 512, device=device)
            face_metadata = []
        
        # Extract body embeddings
        try:
            body_embeddings, body_boxes = self.body_extractor(img_np)
            # Move embeddings to the correct device
            body_embeddings = body_embeddings.to(device)
        except Exception as e:
            print(f"Error in body extraction: {e}")
            # Return zero tensor if body extraction fails
            body_embeddings = torch.zeros(1, 1, 1024, device=device)
            body_boxes = []
        
        # Take only the first/primary face embedding
        # If no faces detected, face_embeddings will already be a zero tensor of shape [1, 1, 512]
        face_embedding = face_embeddings[:, 0:1, :] if face_embeddings.size(1) > 0 else face_embeddings
        
        # Take only the primary face metadata or empty dict if none found
        primary_face_metadata = face_metadata[0] if face_metadata else {}
        
        # Take only the first/primary body embedding
        # If no bodies detected, body_embeddings will already be a zero tensor of shape [1, 1, 1024]
        body_embedding = body_embeddings[:, 0:1, :] if body_embeddings.size(1) > 0 else body_embeddings
        
        # Take only the primary body box or None if none found
        primary_body_box = body_boxes[0] if body_boxes else None
        
        return face_embedding, body_embedding, primary_face_metadata, primary_body_box
    
    def prepare_identity_tokens(self, face_embedding=None, body_embedding=None):
        """
        Prepare identity tokens from face and body embeddings.
        
        Args:
            face_embedding: Tensor of shape [batch_size, 1, face_dim] (single face)
            body_embedding: Tensor of shape [batch_size, 1, body_dim] (single body)
        """
        device = next(self.parameters()).device
        face_tokens = None
        body_tokens = None
        batch_size = 1  # Default batch size
        
        # Process face embedding if provided
        if face_embedding is not None and face_embedding.ndim == 3:
            # Ensure it's on the correct device
            face_embedding = face_embedding.to(device)
            
            batch_size = face_embedding.shape[0]
            face_tokens = self.face_perceiver(face_embedding)
            self.current_face_tokens = face_tokens
        
        # Process body embedding if provided
        if body_embedding is not None and body_embedding.ndim == 3:
            # Ensure it's on the correct device
            body_embedding = body_embedding.to(device)
            
            if batch_size == 1:  # If not set by face, get from body
                batch_size = body_embedding.shape[0]
            body_tokens = self.body_perceiver(body_embedding)
            self.current_body_tokens = body_tokens
            
        # Apply fusion if both tokens are available and fusion is enabled
        if face_tokens is not None and body_tokens is not None and self.use_identity_fusion:
            fused_tokens = self.identity_fusion(face_tokens, body_tokens)
            # Split back to maintain original token counts for each modality
            face_token_count = face_tokens.size(1)
            body_token_count = body_tokens.size(1)
            self.current_face_tokens = fused_tokens[:, :face_token_count, :]
            self.current_body_tokens = fused_tokens[:, face_token_count:, :]
    
    def forward(
        self, 
        *args, 
        face_embedding: Optional[torch.Tensor] = None,
        body_embedding: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass that injects identity information during Flux.1 generation.
        
        Args:
            *args: Arguments to pass to the base model
            face_embedding: Face embedding of shape [B, 1, 512] (single face)
            body_embedding: Body embedding of shape [B, 1, 1024] (single body)
            **kwargs: Keyword arguments to pass to the base model
            
        Returns:
            Output from the base model
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
            
        # Prepare identity tokens for injection
        if face_embedding is not None or body_embedding is not None:
            self.prepare_identity_tokens(face_embedding, body_embedding)
            
        # Forward pass through the base model (hooks will inject identity)
        return self.base_model(*args, **kwargs)
    
    def visualize_attention(self, image_size=(512, 512)):
        """
        Visualize where the identity attention is focused.
        
        Args:
            image_size: Size of the generated image (H, W)
            
        Returns:
            fig: Matplotlib figure with attention visualizations
        """
        # Check if we have attention maps
        if not hasattr(self.face_injection, 'last_attention_map') or self.face_injection.last_attention_map is None:
            print("No face attention maps available. Run inference first.")
            return None
            
        if not hasattr(self.body_injection, 'last_attention_map') or self.body_injection.last_attention_map is None:
            print("No body attention maps available. Run inference first.")
            return None
        
        # Get attention maps
        face_attn = self.face_injection.last_attention_map.detach().cpu()
        body_attn = self.body_injection.last_attention_map.detach().cpu()
        
        # Calculate image dimensions
        H, W = image_size
        seq_len = H * W
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot face attention
        if face_attn.dim() == 3:  # [B, seq_len, num_heads*head_dim]
            face_attn = face_attn.mean(0)  # Average over batch
            face_attn = face_attn.mean(-1)  # Average over tokens
            face_map = face_attn.reshape(H, W)
            axes[0].imshow(face_map, cmap='hot')
            axes[0].set_title("Face Identity Attention")
            axes[0].axis('off')
        
        # Plot body attention
        if body_attn.dim() == 3:
            body_attn = body_attn.mean(0)  # Average over batch
            body_attn = body_attn.mean(-1)  # Average over tokens
            body_map = body_attn.reshape(H, W)
            axes[1].imshow(body_map, cmap='hot')
            axes[1].set_title("Body Identity Attention")
            axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def generate(
        self,
        prompt: str,
        reference_image,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        face_strength: float = 1.0,
        body_strength: float = 1.0,
        **kwargs
    ):
        """
        Generate an image with identity preservation.
        
        Args:
            prompt: Text prompt for image generation
            reference_image: Image to extract identity from (PIL, numpy, or path)
            negative_prompt: Optional negative prompt for generation
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            face_strength: Strength of face identity preservation (0.0-1.0)
            body_strength: Strength of body identity preservation (0.0-1.0)
            **kwargs: Additional keyword arguments for the Flux.1 pipeline
            
        Returns:
            PIL Image or images
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        # Set identity strength
        self.set_identity_strength(face_strength, body_strength)
        
        # Extract identity from reference image - now returns single face/body
        face_embedding, body_embedding, face_metadata, body_box = self.extract_identity(reference_image)
        
        # Create pipeline if not yet available
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up generator if not provided
        if 'generator' not in kwargs and torch.cuda.is_available():
            kwargs['generator'] = torch.Generator("cuda").manual_seed(int(time.time()))
        
        # Set default height and width if not provided
        if 'height' not in kwargs:
            kwargs['height'] = 1024
        if 'width' not in kwargs:
            kwargs['width'] = 1024
        
        # Prepare identity tokens
        self.prepare_identity_tokens(face_embedding, body_embedding)
        
        # Generate with identity preservation
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )
        
        return output.images[0] if hasattr(output, 'images') else output
    
    def batch_generate(
        self,
        prompts: List[str],
        reference_images: List[Union[str, np.ndarray, Image.Image]],
        negative_prompts: Optional[List[str]] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        face_strength: float = 1.0,
        body_strength: float = 1.0,
        **kwargs
    ):
        """
        Generate multiple images with identity preservation using a batch approach.
        
        Args:
            prompts: List of text prompts for image generation
            reference_images: List of images to extract identity from
            negative_prompts: Optional list of negative prompts
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            face_strength: Strength of face identity preservation (0.0-1.0)
            body_strength: Strength of body identity preservation (0.0-1.0)
            **kwargs: Additional keyword arguments for the Flux.1 pipeline
            
        Returns:
            List of PIL Images
        """
        if len(prompts) != len(reference_images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of reference images ({len(reference_images)})")
            
        # Set identity strength
        self.set_identity_strength(face_strength, body_strength)
        
        # Create pipeline if not yet available
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set default height and width if not provided
        if 'height' not in kwargs:
            kwargs['height'] = 1024
        if 'width' not in kwargs:
            kwargs['width'] = 1024
        
        # Process one at a time for consistent handling
        all_images = []
        
        for i, (prompt, ref_img) in enumerate(zip(prompts, reference_images)):
            # Set up generator with different seed for each image
            if torch.cuda.is_available():
                current_generator = torch.Generator("cuda").manual_seed(int(time.time()) + i)
            else:
                current_generator = torch.Generator().manual_seed(int(time.time()) + i)
            
            # Extract identity - single face/body
            face_embedding, body_embedding, _, _ = self.extract_identity(ref_img)
            
            # Prepare identity tokens
            self.prepare_identity_tokens(face_embedding, body_embedding)
            
            # Get negative prompt if available
            neg_prompt = negative_prompts[i] if negative_prompts and i < len(negative_prompts) else None
            
            # Generate with identity preservation
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=current_generator,
                **kwargs
            )
            
            # Collect image
            all_images.append(output.images[0])
        
        return all_images
    
    def add_lora(self, lora_path: str, weight_name: Optional[str] = None, adapter_name: str = "default", lora_scale: float = 0.8):
        """
        Load and apply a LoRA to the Flux.1 model.
        
        Args:
            lora_path: Path to the LoRA file or HF repo
            weight_name: Name of the weight file if lora_path is a directory
            adapter_name: Name of the adapter for the LoRA
            lora_scale: Scale factor for the LoRA weights
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        # Create pipeline if not yet available
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                model=self.base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Move to GPU if needed
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
        
        # Apply LoRA directly to the pipeline
        try:
            self.pipeline.load_lora_weights(
                lora_path, 
                weight_name=weight_name,
                adapter_name=adapter_name
            )
            
            # Set current adapter with weight
            if hasattr(self.pipeline, "set_adapters"):
                self.pipeline.set_adapters([adapter_name], [lora_scale])
                
            # Re-register hooks after loading LoRA
            self._register_hooks()
            
            print(f"Successfully loaded LoRA adapter '{adapter_name}' with scale {lora_scale}")
            
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            print("Make sure the LoRA is compatible with Flux.1")
    
    def add_multiple_loras(self, lora_configs: List[Dict[str, Any]]):
        """
        Load and apply multiple LoRAs to the Flux.1 model.
        
        Args:
            lora_configs: List of dictionaries containing LoRA configurations.
                         Each dict should have: 
                         - 'path': Path to the LoRA file or HF repo
                         - 'weight_name': (Optional) Name of the weight file if path is a directory
                         - 'adapter_name': Name for this adapter
                         - 'scale': Scale factor for this LoRA
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
            
        # Create pipeline if not yet available
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                model=self.base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Move to GPU if needed
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
        
        # Load all LoRAs
        adapter_names = []
        adapter_scales = []
        
        for config in lora_configs:
            try:
                path = config['path']
                adapter_name = config.get('adapter_name', f"lora_{len(adapter_names)}")
                weight_name = config.get('weight_name', None)
                scale = config.get('scale', 1.0)
                
                self.pipeline.load_lora_weights(
                    path, 
                    weight_name=weight_name,
                    adapter_name=adapter_name
                )
                
                adapter_names.append(adapter_name)
                adapter_scales.append(scale)
                
                print(f"Successfully loaded LoRA adapter '{adapter_name}' with scale {scale}")
                
            except Exception as e:
                print(f"Error loading LoRA {config.get('adapter_name', 'unknown')}: {e}")
        
        # Set all adapters with their weights
        if adapter_names and hasattr(self.pipeline, "set_adapters"):
            self.pipeline.set_adapters(adapter_names, adapter_scales)
            
        # Re-register hooks after loading LoRAs
        self._register_hooks()
    
    def fuse_loras(self):
        """
        Fuse loaded LoRAs into the model weights for faster inference.
        """
        if hasattr(self, 'pipeline') and hasattr(self.pipeline, "fuse_lora"):
            self.pipeline.fuse_lora()
            print("LoRAs have been fused into the model")
        else:
            print("No pipeline available or fuse_lora not supported")
        
        # Re-register hooks after fusing
        self._register_hooks()

    def save_visualization(self, save_path: str, image_size=(512, 512)):
        """
        Save attention visualization to a file.
        
        Args:
            save_path: Path to save visualization
            image_size: Size of generated image
        """
        fig = self.visualize_attention(image_size)
        if fig is not None:
            # Ensure the directory exists
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Visualization saved to {save_path}")