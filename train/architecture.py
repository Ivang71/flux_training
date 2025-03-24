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
from torchvision import transforms

from diffusers import FluxPipeline

class PerceiverResampler(nn.Module):
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
        batch_size = x.shape[0]
        
        if x.shape[-1] != self.latents.shape[-1]:
            x = self.input_proj(x)
        
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        for layer, norm in zip(self.layers, self.norm_layers):
            attn_output, _ = layer(
                query=latents,
                key=x,
                value=x
            )
            latents = norm(latents + attn_output)
            
        return latents


class IdentityInjectionLayer(nn.Module):
    def __init__(self, feature_dim: int, identity_dim: int, num_heads: int = 8, input_dims: Optional[List[int]] = None):
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
        
        self.last_attention_map = None
        
        self.img_to_feature = nn.Conv2d(3, feature_dim, kernel_size=1, stride=1, padding=0)
        self.feature_to_img = nn.Conv2d(feature_dim, 3, kernel_size=1, stride=1, padding=0)
        
        self.input_projections = nn.ModuleDict()
        
        standard_dims = [768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 5120]
        
        if input_dims is None:
            input_dims = []
        
        all_dims = list(set(standard_dims + input_dims))
        for dim in all_dims:
            if dim != feature_dim:
                self.input_projections[f"proj_{dim}"] = nn.Linear(dim, feature_dim)
    
    def add_projection_for_dim(self, input_dim: int):
        if input_dim == self.feature_dim:
            return False
            
        proj_key = f"proj_{input_dim}"
        if proj_key in self.input_projections:
            return False
            
        device = next(self.parameters()).device
        self.input_projections[proj_key] = nn.Linear(input_dim, self.feature_dim).to(device)
        return True
    
    def forward(self, hidden_states: torch.Tensor, identity_tokens: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        original_shape = hidden_states.shape
        original_dtype = hidden_states.dtype
        
        is_image_like = len(original_shape) == 4 and original_shape[1] == 3
        
        if is_image_like:
            hidden_states = self.img_to_feature(hidden_states)
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = hidden_states.reshape(B, H*W, C)
        
        elif hidden_states.shape[-1] != self.feature_dim:
            if len(hidden_states.shape) == 3:
                B, seq_len, C = hidden_states.shape
                proj_key = f"proj_{C}"
                if proj_key in self.input_projections:
                    hidden_states = self.input_projections[proj_key](hidden_states)
                else:
                    raise ValueError(f"No projection available for input dimension {C}. "
                                     f"Please add {C} to input_dims when creating the IdentityInjectionLayer.")
        
        if identity_tokens.shape[-1] != self.feature_dim:
            identity_tokens = self.identity_proj(identity_tokens)
        
        identity_tokens = identity_tokens.to(dtype=hidden_states.dtype)
        
        norm_hidden = self.norm1(hidden_states)
        
        chunk_size = 16384
        num_chunks = (norm_hidden.shape[1] + chunk_size - 1) // chunk_size
        
        if num_chunks > 1:
            chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, norm_hidden.shape[1])
                
                chunk = norm_hidden[:, start_idx:end_idx, :]
                attn_output, _ = self.attn(
                    query=chunk,
                    key=identity_tokens,
                    value=identity_tokens
                )
                chunks.append(attn_output)
            
            attn_output = torch.cat(chunks, dim=1)
        else:
            attn_output, attn_weights = self.attn(
                query=norm_hidden,
                key=identity_tokens,
                value=identity_tokens
            )
            self.last_attention_map = attn_weights
        
        hidden_states = hidden_states + (attn_output * strength)
        
        norm_hidden = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + ff_output
        
        if is_image_like:
            B, HW, C = hidden_states.shape
            H = W = int(np.sqrt(HW))
            hidden_states = hidden_states.reshape(B, H, W, C).permute(0, 3, 1, 2)
            hidden_states = self.feature_to_img(hidden_states)
        
        return hidden_states.to(dtype=original_dtype)


class IdentityFusionModule(nn.Module):
    def __init__(self, face_dim=1024, body_dim=1024, hidden_dim=1024):
        super().__init__()
        self.face_dim = face_dim
        self.body_dim = body_dim
        self.hidden_dim = hidden_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
            
    def forward(self, face_tokens, body_tokens):
        face_tokens = self.layer_norm(face_tokens)
        body_tokens = self.layer_norm(body_tokens)
        
        combined = torch.cat([face_tokens, body_tokens], dim=1)
        
        attn_output, _ = self.attention(combined, combined, combined)
        
        output = combined + attn_output
        
        return output


class FaceIdentityExtractor(nn.Module):
    def __init__(self, use_gpu: bool = True):
        super().__init__()
        self.face_analyzer = FaceAnalysis(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider'],
            name="buffalo_l"
        )
        self.face_analyzer.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    
    def forward(self, img: np.ndarray) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        device = next(self.parameters(), torch.tensor(0)).device
        
        faces = self.face_analyzer.get(img)
        
        if not faces:
            return torch.zeros(1, 1, 512, device=device), []
        
        faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        
        primary_face = faces[0]
        face_embedding = torch.tensor(primary_face.embedding, device=device).unsqueeze(0).unsqueeze(0)
        
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
    def __init__(self, model_size: str = "m", confidence: float = 0.25, cache_dir: Optional[str] = None):
        super().__init__()
        self.model = None
        self.model_size = model_size
        self.confidence = confidence
        self.cache_dir = cache_dir or os.path.join(".", "cache", "identity_preserving")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_model()
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = f"yolov8{self.model_size}.pt"
            self.model = YOLO(model_path)
        except ImportError:
            self.model = None
    
    def detect_person(self, img: np.ndarray) -> List[torch.Tensor]:
        if self.model is None:
            self._load_model()
            if self.model is None:
                return []
        
        results = self.model(img, conf=self.confidence, verbose=False)
        
        person_boxes = []
        
        for result in results:
            boxes = result.boxes
            for i, cls in enumerate(boxes.cls):
                if int(cls) == 0:
                    box = boxes.xyxy[i].tolist()
                    score = float(boxes.conf[i])
                    person_boxes.append(torch.tensor([*box, score]))
        
        return person_boxes


class BodyIdentityExtractor(nn.Module):
    def __init__(self, cache_dir: Optional[str] = None, use_gpu: bool = True, yolo_confidence: float = 0.25):
        super().__init__()
        self.cache_dir = cache_dir or os.path.join(".", "cache", "identity_preserving")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.detector = YOLOBodyDetector(confidence=yolo_confidence, cache_dir=self.cache_dir)
        
        self.processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
        self.model = AutoModel.from_pretrained("facebook/dinov2-large")
        
        if use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            
        self.use_cached = False
        self.cached_embeddings = {}
    
    def enable_caching(self, enabled: bool = True):
        self.use_cached = enabled
        if not enabled:
            self.cached_embeddings = {}
    
    def extract_features(self, img: np.ndarray, box: torch.Tensor) -> torch.Tensor:
        device = self.model.device
        
        H, W = img.shape[:2]
        x1, y1, x2, y2 = [max(0, int(coord)) for coord in box[:4]]
        
        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0:
            return torch.zeros(1, 1024, device=device)
        
        if self.use_cached:
            crop_hash = hash(cropped_img.tobytes())
            if crop_hash in self.cached_embeddings:
                return self.cached_embeddings[crop_hash].to(device)
        
        inputs = self.processor(images=cropped_img, return_tensors="pt")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0]
        
        if self.use_cached:
            self.cached_embeddings[crop_hash] = features.detach().clone()
        
        return features
    
    def forward(self, img: np.ndarray) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        device = self.model.device
        
        start_time = time.time()
        
        boxes = self.detector.detect_person(img)
        
        detect_time = time.time()
        print(f"Person detection took: {detect_time - start_time:.2f}s")
        
        if not boxes:
            return torch.zeros(1, 1, 1024, device=device), []
        
        if len(boxes) > 1:
            boxes.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        
        primary_box = boxes[0]
        body_embedding = self.extract_features(img, primary_box).unsqueeze(0)
        
        feature_time = time.time()
        print(f"Feature extraction took: {feature_time - detect_time:.2f}s")
        
        return body_embedding, [primary_box]


class IdentityPreservingFlux(nn.Module):
    def __init__(
        self,
        face_embedding_dim: int = 512,
        body_embedding_dim: int = 1024,
        num_face_latents: int = 16,
        num_body_latents: int = 16,
        num_fused_latents: int = 32,
        face_injection_index: int = 18,
        body_injection_index: int = 10,
        use_gpu: bool = True,
        cache_dir: Optional[str] = None,
        use_identity_fusion: bool = True,
        yolo_confidence: float = 0.25,
        input_dims: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.face_extractor = FaceIdentityExtractor(use_gpu=use_gpu)
        self.body_extractor = BodyIdentityExtractor(
            cache_dir=cache_dir, 
            use_gpu=use_gpu,
            yolo_confidence=yolo_confidence
        )
        
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device_map = "auto" if torch.cuda.is_available() else None
        
        self.face_perceiver = PerceiverResampler(
            input_dim=face_embedding_dim,
            latent_dim=1024,
            num_latents=num_face_latents
        )
        
        self.body_perceiver = PerceiverResampler(
            input_dim=body_embedding_dim,
            latent_dim=1024,
            num_latents=num_body_latents
        )
        
        self.use_identity_fusion = use_identity_fusion
        if use_identity_fusion:
            self.identity_fusion = IdentityFusionModule(face_dim=1024, body_dim=1024)
            self.fused_perceiver = PerceiverResampler(
                input_dim=1024,
                latent_dim=1024,
                num_latents=num_fused_latents
            )
        
        # Default input dimensions if not provided
        if input_dims is None:
            input_dims = [768, 1024, 1280, 2048]  # Common transformer dimensions
        
        self.face_injection = IdentityInjectionLayer(
            feature_dim=1024,
            identity_dim=1024,
            input_dims=input_dims
        )
        
        self.body_injection = IdentityInjectionLayer(
            feature_dim=1024,
            identity_dim=1024,
            input_dims=input_dims
        )
        
        self.face_injection_index = face_injection_index
        self.body_injection_index = body_injection_index
        
        self.base_model = None
        
        self.hooks = {}
        
        self.face_strength = 1.0
        self.body_strength = 1.0
        
        self.training_mode = True
        
        # Register buffers for identity tokens
        self.register_buffer('current_face_tokens', None, persistent=False)
        self.register_buffer('current_body_tokens', None, persistent=False)
    
    def set_training_mode(self, mode=True):
        self.training_mode = mode
        for module in self.children():
            if not hasattr(module, 'train') or not callable(module.train) or isinstance(module, nn.Module):
                if hasattr(module, 'training'):
                    module.training = mode
        
        return self
        
    def train(self, mode=True):
        return self.set_training_mode(mode)
    
    def eval(self):
        return self.set_training_mode(False)
    
    def set_identity_strength(self, face_strength: float = 1.0, body_strength: float = 1.0):
        self.face_strength = max(0.0, min(1.0, face_strength))
        self.body_strength = max(0.0, min(1.0, body_strength))
    
    def _get_flux_module_by_index(self, idx: int) -> nn.Module:
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        try:
            if hasattr(self.base_model, 'transformer_blocks'):
                return self.base_model.transformer_blocks[idx]
            elif hasattr(self.base_model, 'double_stream_blocks') and idx < 19:
                return self.base_model.double_stream_blocks[idx]
            elif hasattr(self.base_model, 'single_stream_blocks') and idx >= 19 and idx < 57:
                return self.base_model.single_stream_blocks[idx - 19]
            elif hasattr(self.base_model, 'blocks'):
                return self.base_model.blocks[idx]
            elif hasattr(self.base_model, 'layers'):
                return self.base_model.layers[idx]
            else:
                print(f"WARNING: Could not find appropriate blocks in the model structure.")
                print(f"Available attributes: {dir(self.base_model)}")
                return self.base_model
        except (IndexError, AttributeError) as e:
            print(f"Error accessing model layer at index {idx}: {e}")
            print(f"Model structure may have changed. Available attributes: {dir(self.base_model)}")
            return self.base_model
    
    def _register_hooks(self):
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        self._unregister_hooks()

        def face_hook(module, input_tensors, output):
            if not hasattr(self, 'current_face_tokens') or self.current_face_tokens is None:
                return output
            
            hidden_states = output
            
            if self.face_strength > 0:
                hidden_states = self.face_injection(
                    hidden_states=hidden_states,
                    identity_tokens=self.current_face_tokens,
                    strength=self.face_strength
                )
            
            return hidden_states

        def body_hook(module, input_tensors, output):
            if not hasattr(self, 'current_body_tokens') or self.current_body_tokens is None:
                return output
            
            hidden_states = output
            
            if self.body_strength > 0:
                hidden_states = self.body_injection(
                    hidden_states=hidden_states,
                    identity_tokens=self.current_body_tokens,
                    strength=self.body_strength
                )
            
            return hidden_states
        
        try:
            self.hooks['face'] = self._get_flux_module_by_index(self.face_injection_index).register_forward_hook(face_hook)
            self.hooks['body'] = self._get_flux_module_by_index(self.body_injection_index).register_forward_hook(body_hook)
            print("Successfully registered identity injection hooks")
        except Exception as e:
            print(f"Failed to register hooks: {e}")
            print("Identity injection may not work properly")
    
    def _unregister_hooks(self):
        for hook_name, hook in self.hooks.items():
            hook.remove()
        self.hooks = {}
    
    def _scan_model_structure(self):
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        print("\nScanning Flux model structure...")
        
        if hasattr(self.base_model, 'transformer_blocks'):
            blocks = self.base_model.transformer_blocks
            print(f"Found transformer_blocks: {len(blocks)} blocks")
            total_blocks = len(blocks)
            self.face_injection_index = int(total_blocks * 0.9)
            self.body_injection_index = int(total_blocks * 0.5)
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
        
        print("WARNING: Could not identify model structure.")
        print(f"Model attributes: {dir(self.base_model)}")
        print("Using default injection indices, but hooks may not work properly.")
        return False
        
    def load_base_model(self, base_model=None):
        if base_model is not None:
            self.base_model = base_model
            self._scan_model_structure()
            self._register_hooks()
            # Scan for dimensions in the model to add necessary projections
            self.scan_model_for_dimensions()
            return self.base_model
            
        try:
            print(f"Loading Flux model with torch_dtype={self.torch_dtype}")
            
            try:
                pipe = FluxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                
                self.base_model = pipe.transformer
                self._scan_model_structure()
                self._register_hooks()
                
                # Scan for dimensions in the model to add necessary projections
                self.scan_model_for_dimensions()
                
                self.pipeline = pipe
                
                print("Successfully loaded Flux model and pipeline")
                return self.base_model
            except ImportError as e:
                print(f"Could not import FluxPipeline from diffusers: {e}")
                print("Make sure you have the latest version installed.")
                
            except Exception as e:
                print(f"Error loading Flux model: {e}")
            
            return None
        except Exception as e:
            print(f"Error in load_base_model: {e}")
            return None
    
    def extract_identity(self, image):
        device = next(self.parameters()).device
        
        if isinstance(image, torch.Tensor) and image.dim() == 4 and image.shape[0] > 1:
            batch_size = image.shape[0]
            
            all_face_embeddings = []
            all_body_embeddings = []
            all_face_metadata = []
            all_body_boxes = []
            
            for i in range(batch_size):
                single_img = image[i:i+1]
                
                single_img_np = single_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                if single_img_np.min() < 0 or single_img_np.max() > 1:
                    mean = np.array([0.5, 0.5, 0.5])
                    std = np.array([0.5, 0.5, 0.5])
                    single_img_np = std * single_img_np + mean
                
                single_img_np = np.clip(single_img_np * 255, 0, 255).astype(np.uint8)
                
                face_emb, body_emb, face_meta, body_box = self._process_identity_from_numpy(single_img_np)
                
                face_emb = face_emb.to(device)
                body_emb = body_emb.to(device)
                
                all_face_embeddings.append(face_emb)
                all_body_embeddings.append(body_emb)
                all_face_metadata.append(face_meta)
                all_body_boxes.append(body_box)
                
            face_embedding = torch.cat(all_face_embeddings, dim=0).to(device)
            body_embedding = torch.cat(all_body_embeddings, dim=0).to(device)
            
            return face_embedding, body_embedding, all_face_metadata, all_body_boxes
        
        if isinstance(image, str):
            from PIL import Image
            image = np.array(Image.open(image).convert('RGB'))
        elif hasattr(image, 'mode'):
            image = np.array(image.convert('RGB'))
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
                
            image = image.permute(1, 2, 0).cpu().numpy()
            
            if image.min() < 0 or image.max() > 1:
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                image = std * image + mean
                
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        face_embedding, body_embedding, face_metadata, body_box = self._process_identity_from_numpy(image)
        face_embedding = face_embedding.to(device)
        body_embedding = body_embedding.to(device) 
        
        return face_embedding, body_embedding, face_metadata, body_box
    
    def _process_identity_from_numpy(self, img_np):
        device = next(self.parameters()).device
        
        try:
            face_embeddings, face_metadata = self.face_extractor(img_np)
            face_embeddings = face_embeddings.to(device)
        except Exception as e:
            print(f"Error in face extraction: {e}")
            face_embeddings = torch.zeros(1, 1, 512, device=device)
            face_metadata = []
        
        try:
            body_embeddings, body_boxes = self.body_extractor(img_np)
            body_embeddings = body_embeddings.to(device)
        except Exception as e:
            print(f"Error in body extraction: {e}")
            body_embeddings = torch.zeros(1, 1, 1024, device=device)
            body_boxes = []
        
        face_embedding = face_embeddings[:, 0:1, :] if face_embeddings.size(1) > 0 else face_embeddings
        
        primary_face_metadata = face_metadata[0] if face_metadata else {}
        
        body_embedding = body_embeddings[:, 0:1, :] if body_embeddings.size(1) > 0 else body_embeddings
        
        primary_body_box = body_boxes[0] if body_boxes else None
        
        return face_embedding, body_embedding, primary_face_metadata, primary_body_box
    
    def prepare_identity_tokens(self, face_embedding=None, body_embedding=None):
        device = next(self.parameters()).device
        face_tokens = None
        body_tokens = None
        batch_size = 1
        
        if face_embedding is not None and face_embedding.ndim == 3:
            face_embedding = face_embedding.to(device)
            
            batch_size = face_embedding.shape[0]
            face_tokens = self.face_perceiver(face_embedding)
            # Create buffer if it doesn't exist
            if self.current_face_tokens is None or self.current_face_tokens.shape != face_tokens.shape:
                self.register_buffer('current_face_tokens', face_tokens, persistent=False)
            else:
                self.current_face_tokens.copy_(face_tokens)
        
        if body_embedding is not None and body_embedding.ndim == 3:
            body_embedding = body_embedding.to(device)
            
            if batch_size == 1:
                batch_size = body_embedding.shape[0]
            body_tokens = self.body_perceiver(body_embedding)
            # Create buffer if it doesn't exist
            if self.current_body_tokens is None or self.current_body_tokens.shape != body_tokens.shape:
                self.register_buffer('current_body_tokens', body_tokens, persistent=False)
            else:
                self.current_body_tokens.copy_(body_tokens)
            
        if face_tokens is not None and body_tokens is not None and self.use_identity_fusion:
            fused_tokens = self.identity_fusion(face_tokens, body_tokens)
            face_token_count = face_tokens.size(1)
            body_token_count = body_tokens.size(1)
            # Update the buffers with fused tokens
            self.current_face_tokens.copy_(fused_tokens[:, :face_token_count, :])
            self.current_body_tokens.copy_(fused_tokens[:, face_token_count:, :])
    
    def __call__(self, input_images, prompt=None, negative_prompt=None, num_inference_steps=30, guidance_scale=7.5, **kwargs):
        if self.training_mode:
            return input_images
        
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
        if prompt is None:
            if isinstance(input_images, torch.Tensor) and input_images.dim() > 3:
                batch_size = input_images.shape[0]
                prompt = ["A high quality photo"] * batch_size
            else:
                prompt = "A high quality photo"
        
        with torch.no_grad():
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            )
            
            if hasattr(output, 'images'):
                output_tensor = []
                for img in output.images:
                    img_tensor = transforms.ToTensor()(img).unsqueeze(0)
                    output_tensor.append(img_tensor)
                return torch.cat(output_tensor, dim=0).to(input_images.device)
            
            return output
    
    def visualize_attention(self, image_size=(512, 512)):
        if not hasattr(self.face_injection, 'last_attention_map') or self.face_injection.last_attention_map is None:
            print("No face attention maps available. Run inference first.")
            return None
            
        if not hasattr(self.body_injection, 'last_attention_map') or self.body_injection.last_attention_map is None:
            print("No body attention maps available. Run inference first.")
            return None
        
        face_attn = self.face_injection.last_attention_map.detach().cpu()
        body_attn = self.body_injection.last_attention_map.detach().cpu()
        
        H, W = image_size
        seq_len = H * W
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        if face_attn.dim() == 3:
            face_attn = face_attn.mean(0)
            face_attn = face_attn.mean(-1)
            face_map = face_attn.reshape(H, W)
            axes[0].imshow(face_map, cmap='hot')
            axes[0].set_title("Face Identity Attention")
            axes[0].axis('off')
        
        if body_attn.dim() == 3:
            body_attn = body_attn.mean(0)
            body_attn = body_attn.mean(-1)
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
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        self.set_identity_strength(face_strength, body_strength)
        
        face_embedding, body_embedding, face_metadata, body_box = self.extract_identity(reference_image)
        
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        if 'generator' not in kwargs and torch.cuda.is_available():
            kwargs['generator'] = torch.Generator("cuda").manual_seed(int(time.time()))
        
        if 'height' not in kwargs:
            kwargs['height'] = 1024
        if 'width' not in kwargs:
            kwargs['width'] = 1024
        
        self.prepare_identity_tokens(face_embedding, body_embedding)
        
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
        if len(prompts) != len(reference_images):
            raise ValueError(f"Number of prompts ({len(prompts)}) must match number of reference images ({len(reference_images)})")
            
        self.set_identity_strength(face_strength, body_strength)
        
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        if 'height' not in kwargs:
            kwargs['height'] = 1024
        if 'width' not in kwargs:
            kwargs['width'] = 1024
        
        all_images = []
        
        for i, (prompt, ref_img) in enumerate(zip(prompts, reference_images)):
            if torch.cuda.is_available():
                current_generator = torch.Generator("cuda").manual_seed(int(time.time()) + i)
            else:
                current_generator = torch.Generator().manual_seed(int(time.time()) + i)
            
            face_embedding, body_embedding, _, _ = self.extract_identity(ref_img)
            
            self.prepare_identity_tokens(face_embedding, body_embedding)
            
            neg_prompt = negative_prompts[i] if negative_prompts and i < len(negative_prompts) else None
            
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=current_generator,
                **kwargs
            )
            
            all_images.append(output.images[0])
        
        return all_images
    
    def add_lora(self, lora_path: str, weight_name: Optional[str] = None, adapter_name: str = "default", lora_scale: float = 0.8):
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
        
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                model=self.base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
        
        try:
            self.pipeline.load_lora_weights(
                lora_path, 
                weight_name=weight_name,
                adapter_name=adapter_name
            )
            
            if hasattr(self.pipeline, "set_adapters"):
                self.pipeline.set_adapters([adapter_name], [lora_scale])
                
            self._register_hooks()
            
            print(f"Successfully loaded LoRA adapter '{adapter_name}' with scale {lora_scale}")
            
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            print("Make sure the LoRA is compatible with Flux.1")
    
    def add_multiple_loras(self, lora_configs: List[Dict[str, Any]]):
        if self.base_model is None:
            raise ValueError("Base model not loaded. Call load_base_model first.")
            
        if not hasattr(self, 'pipeline'):
            self.pipeline = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                model=self.base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
        
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
        
        if adapter_names and hasattr(self.pipeline, "set_adapters"):
            self.pipeline.set_adapters(adapter_names, adapter_scales)
            
        self._register_hooks()
    
    def fuse_loras(self):
        if hasattr(self, 'pipeline') and hasattr(self.pipeline, "fuse_lora"):
            self.pipeline.fuse_lora()
            print("LoRAs have been fused into the model")
        else:
            print("No pipeline available or fuse_lora not supported")
        
        self._register_hooks()

    def save_visualization(self, save_path: str, image_size=(512, 512)):
        fig = self.visualize_attention(image_size)
        if fig is not None:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Visualization saved to {save_path}")
    
    def verify_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        modules_with_params = {}
        for name, module in self.named_modules():
            if list(module.parameters(recurse=False)):
                param_count = sum(p.numel() for p in module.parameters(recurse=False))
                modules_with_params[name] = param_count
        
        device_transfer_success = True
        original_device = next(self.parameters()).device
        
        try:
            cpu_model = self.to("cpu")
            cpu_device = next(cpu_model.parameters()).device
            device_transfer_success = str(cpu_device) == "cpu"
            
            self.to(original_device)
        except Exception as e:
            device_transfer_success = False
            print(f"Failed to transfer model to CPU: {e}")
        
        buffers = {name: buffer.shape for name, buffer in self.named_buffers()}
        
        hooks_registered = len(self.hooks) > 0
        
        dynamic_modules = []
        for name, module in self.named_modules():
            if isinstance(module, IdentityInjectionLayer) and hasattr(module, 'input_projections'):
                dynamic_keys = list(module.input_projections.keys())
                if dynamic_keys:
                    dynamic_modules.append(f"{name}: {dynamic_keys}")
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "modules_with_parameters": modules_with_params,
            "device_transfer_successful": device_transfer_success,
            "registered_buffers": buffers,
            "hooks_registered": hooks_registered,
            "dynamic_modules": dynamic_modules
        }

    def add_projection_for_dimension(self, input_dim: int) -> bool:
        added = False
        if hasattr(self, 'face_injection'):
            added = self.face_injection.add_projection_for_dim(input_dim) or added
            
        if hasattr(self, 'body_injection'):
            added = self.body_injection.add_projection_for_dim(input_dim) or added
            
        return added
    
    def scan_model_for_dimensions(self) -> List[int]:
        if self.base_model is None:
            print("Base model not loaded. Cannot scan for dimensions.")
            return []
            
        found_dims = []
        
        def check_module_dims(module):
            dims = []
            for name, param in module.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    if param.shape[0] not in dims and param.shape[0] != 1024:
                        dims.append(param.shape[0])
                    if param.shape[1] not in dims and param.shape[1] != 1024:
                        dims.append(param.shape[1])
            return dims
        
        for name, module in self.base_model.named_modules():
            dims = check_module_dims(module)
            for dim in dims:
                if dim > 64 and dim not in found_dims:
                    found_dims.append(dim)
                    self.add_projection_for_dimension(dim)
        
        print(f"Found and added support for {len(found_dims)} dimensions: {found_dims}")
        return found_dims

def test_model_device_transfer(model_class=IdentityPreservingFlux, **kwargs):
    print("Testing model device transfer...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, testing CPU transfer only")
        
    model = model_class(**kwargs).to("cpu")
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial parameter count: {initial_params}")
    
    buffer_count = sum(1 for _ in model.buffers())
    print(f"Buffer count: {buffer_count}")
    
    all_params_on_cpu = all(p.device.type == "cpu" for p in model.parameters())
    print(f"All parameters on CPU: {all_params_on_cpu}")
    
    results = {
        "model_class": model_class.__name__,
        "initial_params": initial_params,
        "buffer_count": buffer_count,
        "all_params_on_cpu": all_params_on_cpu,
        "cuda_tests": {}
    }
    
    if torch.cuda.is_available():
        try:
            model.to("cuda")
            cuda_param_count = sum(p.numel() for p in model.parameters())
            all_params_on_cuda = all(p.device.type == "cuda" for p in model.parameters())
            cuda_buffer_count = sum(1 for _ in model.buffers())
            
            print(f"CUDA parameter count: {cuda_param_count}")
            print(f"All parameters on CUDA: {all_params_on_cuda}")
            print(f"CUDA buffer count: {cuda_buffer_count}")
            
            params_match = cuda_param_count == initial_params
            buffers_match = cuda_buffer_count == buffer_count
            
            results["cuda_tests"] = {
                "cuda_param_count": cuda_param_count,
                "all_params_on_cuda": all_params_on_cuda,
                "cuda_buffer_count": cuda_buffer_count,
                "params_match": params_match,
                "buffers_match": buffers_match
            }
            
            model.to("cpu")
            back_to_cpu_params = sum(p.numel() for p in model.parameters())
            all_back_on_cpu = all(p.device.type == "cpu" for p in model.parameters())
            
            print(f"Back to CPU parameter count: {back_to_cpu_params}")
            print(f"All parameters back on CPU: {all_back_on_cpu}")
            
            results["back_to_cpu"] = {
                "back_to_cpu_params": back_to_cpu_params,
                "all_back_on_cpu": all_back_on_cpu,
                "params_match": back_to_cpu_params == initial_params
            }
            
        except Exception as e:
            print(f"Error during CUDA transfer: {e}")
            results["cuda_tests"]["error"] = str(e)
    
    if hasattr(model, 'verify_parameters'):
        verification_results = model.verify_parameters()
        results["verification_results"] = verification_results
    
    try:
        dummy_input = torch.randn(1, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            _ = model.extract_identity(dummy_input)
        results["dummy_input_test"] = "passed"
    except Exception as e:
        print(f"Error during dummy input test: {e}")
        results["dummy_input_test"] = f"failed: {str(e)}"
    
    print("Device transfer tests completed.")
    return results

def test_no_dynamic_instantiation():
    print("Testing that no dynamic instantiation occurs in forward method...")
    
    model = IdentityPreservingFlux(input_dims=[512, 1024]).to("cpu")
    
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial parameter count: {initial_params}")
    
    initial_projection_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            initial_projection_layers[name] = len(module.input_projections)
    
    print(f"Initial projection layers: {initial_projection_layers}")
    
    dummy_input = torch.randn(1, 3, 32, 32)
    identity_tokens = torch.randn(1, 4, 1024)
    
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            try:
                _ = module.forward(dummy_input, identity_tokens)
                print(f"Forward pass succeeded for {name} with image input")
            except Exception as e:
                print(f"Error in forward pass for {name} with image input: {e}")
    
    feature_input = torch.randn(1, 10, 1024)
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            try:
                _ = module.forward(feature_input, identity_tokens)
                print(f"Forward pass succeeded for {name} with 1024-dim features")
            except Exception as e:
                print(f"Error in forward pass for {name} with 1024-dim features: {e}")
    
    feature_input_512 = torch.randn(1, 10, 512)
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            try:
                _ = module.forward(feature_input_512, identity_tokens)
                print(f"Forward pass succeeded for {name} with 512-dim features")
            except Exception as e:
                print(f"Error in forward pass for {name} with 512-dim features: {e}")
    
    feature_input_768 = torch.randn(1, 10, 768)
    expected_errors = {}
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            try:
                _ = module.forward(feature_input_768, identity_tokens)
                print(f"WARNING: Forward pass succeeded for {name} with unsupported 768-dim features")
                expected_errors[name] = False
            except ValueError as e:
                print(f"Expected error for {name} with unsupported 768-dim features: {e}")
                expected_errors[name] = True
            except Exception as e:
                print(f"Unexpected error type for {name}: {e}")
                expected_errors[name] = False
    
    final_params = sum(p.numel() for p in model.parameters())
    params_unchanged = initial_params == final_params
    print(f"Parameters unchanged: {params_unchanged} ({initial_params} vs {final_params})")
    
    final_projection_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            final_projection_layers[name] = len(module.input_projections)
    
    projections_unchanged = initial_projection_layers == final_projection_layers
    print(f"Projection layers unchanged: {projections_unchanged}")
    print(f"Initial: {initial_projection_layers}")
    print(f"Final: {final_projection_layers}")
    
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            module.add_projection_for_dim(768)
    
    for name, module in model.named_modules():
        if isinstance(module, IdentityInjectionLayer):
            try:
                _ = module.forward(feature_input_768, identity_tokens)
                print(f"Forward pass now succeeds for {name} with 768-dim features after adding projection")
            except Exception as e:
                print(f"Error in forward pass for {name} with 768-dim features after adding projection: {e}")
    
    added_params = sum(p.numel() for p in model.parameters())
    params_increased = added_params > final_params
    print(f"Parameters increased after adding projection: {params_increased} ({final_params} -> {added_params})")
    
    return {
        "initial_params": initial_params,
        "final_params": final_params,
        "added_params": added_params,
        "params_unchanged_during_forward": params_unchanged,
        "projections_unchanged_during_forward": projections_unchanged,
        "expected_errors_raised": expected_errors,
        "params_increased_after_adding_projection": params_increased
    }

if __name__ == "__main__":
    # Test that no dynamic instantiation happens
    no_dynamic_instantiation_results = test_no_dynamic_instantiation()
    print("\nNo Dynamic Instantiation Test Results:", no_dynamic_instantiation_results)
    
    # Original device transfer test
    test_results = test_model_device_transfer(
        input_dims=[768, 1024, 1280, 2048]
    )
    print("\nDevice Transfer Test Results:", test_results)