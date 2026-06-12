"""
dataset_hdr.py — Radiance HDR Training Dataset
════════════════════════════════════════════════════════════════════════════════

Builds (latent, log_target) training pairs for the RadianceTurboDecoder
using the existing Radiance encode pipeline.

Pipeline per sample:
    scene-linear EXR
        → _prepare_for_vae(hdr_mode="Compress (Log)")   [color.py / vae.py]
        → VAE encoder → latent z
        → stored pair: (z, log_coded_image)

The decoder is trained to predict log_coded_image from z.
The existing decode path (soft-shoulder + inverse log) then converts
log_coded → scene-linear at inference time.

SUPPORTED DATA SOURCES:
  1. Local EXR directories (OpenEXR or imageio)
  2. HDR Haven / Poly Haven layout (recursive scan for .exr / .hdr)
  3. Pre-generated pairs from a previous run (fast reload)

REQUIREMENTS:
  pip install openexr imageio imageio-ffmpeg tqdm  (all in requirements.txt)
  pip install opencv-python                         (nodes_io.py dep)
"""

import os
import json
import logging
import math
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("radiance.dataset_hdr")

# ─────────────────────────────────────────────────────────────────────────────
#  EXR / HDR loading
# ─────────────────────────────────────────────────────────────────────────────

_OPENEXR_AVAILABLE = False
try:
    import OpenEXR
    import Imath
    _OPENEXR_AVAILABLE = True
except ImportError:
    pass


def load_exr_as_linear(path: str) -> Optional[np.ndarray]:
    """
    Load an EXR or HDR file as scene-linear float32 (H, W, 3).
    Returns None if the file cannot be read.

    Priority:
      1. OpenEXR Python binding (handles multi-part, deep, arbitrary channels)
      2. imageio v3 (handles .hdr / Radiance HDR format)
      3. OpenCV (fallback for simple 3-channel EXRs, and 16-bit linear PNGs)
    """
    if path.lower().endswith(".png"):
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("cv2 returned None for PNG")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Revert the prepare_training_data.py 16-bit PNG normalization
            # The PNG was saved as np.clip(linear * (203.0 / 10000.0), 0, 1) * 65535.0
            img = img.astype(np.float32) / 65535.0
            return img * (10000.0 / 203.0)
        except Exception as e:
            logger.debug(f"PNG loader failed for {path}: {e}")
            return None

    try:
        if _OPENEXR_AVAILABLE and path.lower().endswith(".exr"):
            return _load_exr_openexr(path)
    except Exception as e:
        logger.debug(f"OpenEXR failed for {path}: {e}")

    try:
        import imageio.v3 as iio
        arr = iio.imread(path, plugin="EXR-FI" if path.lower().endswith(".exr") else None)
        if arr is None:
            raise ValueError("imageio returned None")
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr
    except Exception as e:
        logger.debug(f"imageio failed for {path}: {e}")

    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError("cv2 returned None")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32)
    except Exception as e:
        logger.debug(f"cv2 failed for {path}: {e}")

    logger.warning(f"All loaders failed for: {path}")
    return None


def _load_exr_openexr(path: str) -> np.ndarray:
    """Load 3-channel EXR using the OpenEXR binding."""
    f = OpenEXR.InputFile(path)
    header = f.header()
    dw = header["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1

    # Read first 3 channels (prefer R/G/B over Y/BY/RY)
    channels = list(header["channels"].keys())
    rgb_ch = []
    for preferred in (["R", "G", "B"], ["r", "g", "b"]):
        if all(c in channels for c in preferred):
            rgb_ch = preferred
            break
    if not rgb_ch:
        rgb_ch = channels[:3]

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    arrays = []
    for ch in rgb_ch:
        raw = f.channel(ch, pt)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        arrays.append(arr)
    return np.stack(arrays, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  EXR → Log-coded image using the Radiance pipeline
# ─────────────────────────────────────────────────────────────────────────────

def linear_to_log_coded(
    image_linear: np.ndarray,
    log_curve: str = "ARRI LogC4",
) -> np.ndarray:
    """
    Apply the Radiance log compression curve to a scene-linear image.

    This mirrors the _prepare_for_vae() Compress (Log) path in vae.py so
    that training targets are in the same domain as the encoded VAE input.

    Returns log-coded float32 in approximately [0, 1.08].
    """
    # Use the package's self-contained, round-trip-exact curves so the encode
    # here is the exact inverse of the decode in rudra.normalization /
    # rudra.color_curves used at training and inference time (review §3.1).
    try:
        from rudra.color_curves import encode_linear, CURVE_NAME_TO_KEY
    except ImportError:
        from color_curves import encode_linear, CURVE_NAME_TO_KEY  # type: ignore

    curve_key = CURVE_NAME_TO_KEY.get(log_curve, "logc4")

    # Clamp negatives (log curves are undefined for x < 0 except small floor)
    image_linear = np.clip(image_linear, -0.03, None)

    t = torch.from_numpy(image_linear.astype(np.float32))
    log_coded = encode_linear(t, curve_key).numpy()
    return log_coded.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  HDR Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HDRPairDataset(Dataset):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    Dataset of (latent, log_coded_target) pairs for RadianceTurboDecoder training.

    Can operate in two modes:
      ONLINE  — loads EXR files at training time, encodes on-the-fly.
                Slower, requires VAE available during training.
      OFFLINE — loads pre-generated .npz pair files (fast, no VAE needed).
                Generate pairs with HDRPairDataset.generate_pairs().

    Args:
        pair_dir:   Directory of .npz pair files (offline mode).
        exr_dirs:   List of directories with EXR/HDR files (online mode).
        vae:        ComfyUI VAE object (required for online mode).
        image_size: Target spatial size (width, height) — images are
                    centre-cropped and resized to this.
        log_curve:  Log compression curve to apply.
        augment:    Enable horizontal flip augmentation.
    """

    EXTENSIONS = {".exr", ".hdr", ".EXR", ".HDR", ".png", ".PNG"}

    def __init__(
        self,
        pair_dir: Optional[str] = None,
        exr_dirs: Optional[List[str]] = None,
        vae=None,
        image_size: Tuple[int, int] = (512, 512),
        log_curve: str = "ARRI LogC4",
        augment: bool = True,
        pairs: Optional[List[Path]] = None,
    ):
        self.image_size = image_size
        self.log_curve = log_curve
        self.augment = augment
        self.mode = "offline" if pair_dir else "online"

        if self.mode == "offline":
            self.pairs = pairs if pairs is not None else sorted(Path(pair_dir).glob("*.npz"))
            if not self.pairs:
                raise FileNotFoundError(
                    f"No .npz pair files found in: {pair_dir}\n"
                    f"Generate them first with HDRPairDataset.generate_pairs()"
                )
            logger.info(f"[HDR Dataset] Offline mode: {len(self.pairs)} pairs in {pair_dir}")

        else:  # online
            if not exr_dirs:
                raise ValueError("exr_dirs required for online mode")
            if vae is None:
                raise ValueError("vae required for online mode")
            self.vae = vae
            self.exr_paths = self._scan_exr_dirs(exr_dirs)
            if not self.exr_paths:
                raise FileNotFoundError(
                    f"No EXR/HDR files found in: {exr_dirs}"
                )
            logger.info(
                f"[HDR Dataset] Online mode: {len(self.exr_paths)} EXR files "
                f"from {len(exr_dirs)} directories"
            )

    def __len__(self) -> int:
        if self.mode == "offline":
            return len(self.pairs)
        return len(self.exr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "offline":
            return self._load_pair(self.pairs[idx])
        return self._encode_online(self.exr_paths[idx])

    # ── Offline: load pre-generated .npz pair ─────────────────────────────────

    def _load_pair(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(str(path))
        latent = torch.from_numpy(data["latent"].astype(np.float32))
        target = torch.from_numpy(data["log_coded"].astype(np.float32))

        if self.augment and torch.rand(1).item() > 0.5:
            latent = torch.flip(latent, dims=[-1])
            target = torch.flip(target, dims=[-1])

        return latent, target

    # ── Online: load EXR, encode with VAE ─────────────────────────────────────

    def _encode_online(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        img_linear = load_exr_as_linear(path)
        if img_linear is None:
            # Return a dummy pair on load failure (prevents DataLoader crash)
            W, H = self.image_size
            return (
                torch.zeros(16, H // 8, W // 8),
                torch.zeros(H, W, 3),
            )

        img_linear = self._preprocess(img_linear)
        log_coded  = linear_to_log_coded(img_linear, self.log_curve)

        # Encode log-coded image with VAE
        t = torch.from_numpy(log_coded).unsqueeze(0)  # (1, H, W, 3)
        with torch.no_grad():
            latent_dict = self.vae.encode(t.permute(0, 3, 1, 2))
            latent = latent_dict["samples"].squeeze(0)   # (C, H//8, W//8)

        if self.augment and torch.rand(1).item() > 0.5:
            latent    = torch.flip(latent,                   dims=[-1])
            log_coded = np.flip(log_coded, axis=1).copy()

        return latent, torch.from_numpy(log_coded)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _preprocess(self, img: np.ndarray, random_crop: bool = False) -> np.ndarray:
        """Crop and resize to self.image_size. If random_crop=True, picks a random valid window."""
        W_tgt, H_tgt = self.image_size
        H_src, W_src = img.shape[:2]

        aspect_tgt = W_tgt / H_tgt
        
        if random_crop:
            # Pick a random crop that fits the target aspect ratio
            # For simplicity, we choose a crop height and derive width
            # We want the crop to be at least target size but can be larger for better quality resize
            min_side = min(H_src, int(W_src / aspect_tgt))
            crop_h = np.random.randint(H_tgt, min_side + 1)
            crop_w = int(crop_h * aspect_tgt)
            
            y0 = np.random.randint(0, H_src - crop_h + 1)
            x0 = np.random.randint(0, W_src - crop_w + 1)
            img = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
        else:
            # Centre crop to target aspect ratio (original behavior)
            aspect_src = W_src / H_src
            if aspect_src > aspect_tgt:
                new_w = int(H_src * aspect_tgt)
                x0 = (W_src - new_w) // 2
                img = img[:, x0 : x0 + new_w]
            else:
                new_h = int(W_src / aspect_tgt)
                y0 = (H_src - new_h) // 2
                img = img[y0 : y0 + new_h]

        # Resize (log-safe: operate in linear space)
        import cv2
        img = cv2.resize(img, (W_tgt, H_tgt), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)

    def _scan_exr_dirs(self, dirs: List[str]) -> List[str]:
        paths = []
        for d in dirs:
            for root, _, files in os.walk(d):
                for f in files:
                    if Path(f).suffix in self.EXTENSIONS:
                        paths.append(os.path.join(root, f))
        return sorted(paths)

    def split_by_scene(
        self,
        val_split: float = 0.05,
        seed: int = 42
    ) -> Tuple[List[Path], List[Path]]:
        """
        Groups all pairs by their scene ID (to prevent data leakage from crops or video sequences)
        and splits them into train and validation sets based on the scene groups.
        """
        import json
        import random
        from collections import defaultdict
        
        if not self.pairs:
            return [], []
            
        cache_path = Path(self.pairs[0].parent) / ".metadata_cache.json"
        
        file_to_scene = {}
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    file_to_scene = json.load(f)
                logger.info(f"[HDR Dataset] Loaded scene metadata cache from {cache_path}")
            except Exception as e:
                logger.warning(f"[HDR Dataset] Failed to read metadata cache: {e}")
                
        # If cache is missing or incomplete, build it
        missing_pairs = [p for p in self.pairs if p.name not in file_to_scene]
        if missing_pairs:
            logger.info(f"[HDR Dataset] Parsing metadata for {len(missing_pairs)} pairs (lazy-loading)...")
            for p in missing_pairs:
                try:
                    data = np.load(str(p))
                    meta = json.loads(str(data["meta"]))
                    source = meta.get("source", "")
                    
                    # Determine scene ID
                    filename = os.path.basename(source)
                    name, _ = os.path.splitext(filename)
                    parts = name.split("_")
                    if len(parts) > 2 and parts[0] == "tif":
                        scene_id = "_".join(parts[:-1])
                    else:
                        scene_id = name
                        
                    file_to_scene[p.name] = scene_id
                except Exception as e:
                    file_to_scene[p.name] = p.name  # Fallback to unique filename as scene
                    
            # Save the updated cache
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(file_to_scene, f, indent=2)
                logger.info(f"[HDR Dataset] Saved scene metadata cache to {cache_path}")
            except Exception as e:
                logger.warning(f"[HDR Dataset] Failed to save metadata cache: {e}")
                
        # Group pairs by scene ID
        scene_to_pairs = defaultdict(list)
        for p in self.pairs:
            scene_id = file_to_scene.get(p.name, p.name)
            scene_to_pairs[scene_id].append(p)
            
        # Shuffle scenes deterministically
        scenes = sorted(list(scene_to_pairs.keys()))
        rng = random.Random(seed)
        rng.shuffle(scenes)
        
        # Calculate split index
        n_scenes = len(scenes)
        n_val_scenes = max(1, int(n_scenes * val_split))
        
        val_scenes = set(scenes[:n_val_scenes])
        
        train_pairs = []
        val_pairs = []
        
        for scene_id, pairs in scene_to_pairs.items():
            if scene_id in val_scenes:
                val_pairs.extend(pairs)
            else:
                train_pairs.extend(pairs)
                
        logger.info(
            f"[HDR Dataset] Scene-aware split: {len(scenes)} total scenes. "
            f"Train: {len(scenes) - len(val_scenes)} scenes ({len(train_pairs)} pairs). "
            f"Val: {len(val_scenes)} scenes ({len(val_pairs)} pairs)."
        )
        
        return train_pairs, val_pairs

    # ── Static: generate offline pair files ───────────────────────────────────

    @staticmethod
    def generate_pairs(
        exr_dirs: List[str],
        output_dir: str,
        vae,
        image_size: Tuple[int, int] = (512, 512),
        log_curve: str = "ARRI LogC4",
        max_samples: int = -1,
        device: str = "cuda",
        crops_per_image: int = 1,
        target_count: int = -1,
    ) -> int:
        """
        Pre-encode EXR files into (latent, log_coded) .npz pairs.

        Stores each pair as:
            {output_dir}/{hash}.npz
                latent    : float32 (C, H//8, W//8)
                log_coded : float32 (H, W, 3)
                meta      : JSON string (path, log_curve, image_size)

        Returns number of pairs successfully generated.
        """
        from tqdm import tqdm

        os.makedirs(output_dir, exist_ok=True)
        dataset_tmp = HDRPairDataset.__new__(HDRPairDataset)
        dataset_tmp.image_size = image_size
        dataset_tmp.log_curve  = log_curve
        dataset_tmp.augment    = False
        dataset_tmp.vae        = vae
        dataset_tmp.exr_paths  = dataset_tmp._scan_exr_dirs(exr_dirs)
        dataset_tmp.mode       = "online"

        paths = dataset_tmp.exr_paths
        if max_samples > 0:
            paths = paths[:max_samples]

        # ── Scaling Logic ──
        if target_count > 0:
            n_sources = len(paths)
            if n_sources == 0:
                logger.error("No source images found!")
                return 0
            
            crops_per_image = max(1, math.ceil(target_count / n_sources))
            logger.info(
                f"[HDR Dataset] Auto-scaling: target={target_count}, sources={n_sources} "
                f"→ Calculated {crops_per_image} crops per image"
            )

        logger.info(
            f"[HDR Dataset] Generating total possible pairs: {len(paths) * crops_per_image} → {output_dir}"
        )

        n_ok = 0
        for path in tqdm(paths, desc="Encoding EXR pairs"):
            try:
                img_linear_full = load_exr_as_linear(path)
                if img_linear_full is None:
                    continue

                for c_idx in range(crops_per_image):
                    # Random crop if more than 1 requested
                    use_random = (crops_per_image > 1)
                    img_linear = dataset_tmp._preprocess(img_linear_full, random_crop=use_random)
                    log_coded  = linear_to_log_coded(img_linear, dataset_tmp.log_curve)

                    # Encode log-coded image with VAE
                    t = torch.from_numpy(log_coded).unsqueeze(0).to(device)
                    # ComfyUI VAE usually expects (B, H, W, C), but some implementations 
                    # expect (B, C, H, W). We try to match Comfy's standard (B, H, W, C).
                    with torch.no_grad():
                        res = dataset_tmp.vae.encode(t)
                        # Handle both dictionary and raw tensor returns
                        if isinstance(res, dict):
                            latent = res.get("samples", res.get("latent", res))
                        else:
                            latent = res
                        
                        if isinstance(latent, torch.Tensor):
                            latent = latent.squeeze(0).cpu()
                            # If latent is 4D (C, T, H, W) from a 3D VAE like Wan, squeeze out the time dimension if it's 1
                            if latent.ndim == 4 and latent.shape[1] == 1:
                                latent = latent.squeeze(1)
                    # Add random flip augmentation to pairs
                    if np.random.rand() > 0.5:
                        latent = torch.flip(latent, dims=[-1])
                        log_coded = np.flip(log_coded, axis=1).copy()

                    # Deterministic filename from content hash + crop index
                    h = hashlib.md5(f"{path}_{c_idx}".encode()).hexdigest()[:12]
                    out_path = os.path.join(output_dir, f"{h}.npz")

                    np.savez_compressed(
                        out_path,
                        latent=latent.to(torch.float32).numpy(),
                        log_coded=log_coded if isinstance(log_coded, np.ndarray) else log_coded.numpy(),
                        meta=json.dumps({
                            "source": path,
                            "crop_idx": c_idx,
                            "log_curve": log_curve,
                            "image_size": list(image_size),
                        }),
                    )
                    n_ok += 1
            except Exception as e:
                logger.warning(f"Failed to process {path}: {e}")

        logger.info(f"[HDR Dataset] Generated {n_ok} pairs successfully.")
        return n_ok


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone VAE Loader (for CLI usage)
# ─────────────────────────────────────────────────────────────────────────────

def load_vae_standalone(path: str, model_type: str = "flux", device: str = "cuda"):
    """
    Load a VAE checkpoint (.safetensors/.ckpt) without full ComfyUI.
    Supports all models via the unified MODEL_VAE_CONFIG.
    """
    logger.info(f"[VAE Loader] Loading {model_type} VAE from: {path}")
    
    # 1. Load weights if file exists
    state_dict = None
    if path and os.path.isfile(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu")
    else:
        logger.warning(f"[VAE Loader] VAE path '{path}' is not a valid file on disk. Proceeding with state_dict = None.")
    
    # 2. Strip 'vae.' prefix if present
    if state_dict is not None and any(k.startswith("vae.") for k in state_dict):
        state_dict = {k[4:] if k.startswith("vae.") else k: v for k, v in state_dict.items()}

    # 3. Create a mock VAE object that matches the ComfyUI interface
    class StandaloneVAE:
        CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
        def __init__(self, sd, m_type, dev):
            # comfy.sd.VAE expects a torch.device; a bare "cuda" string later
            # triggers `'str' object has no attribute 'type'` during encode.
            self.device = torch.device(dev) if isinstance(dev, str) else dev
            self.model_type = m_type
            
            # Try ComfyUI core first (best compatibility)
            import sys
            comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if comfy_path not in sys.path:
                sys.path.append(comfy_path)
            
            if sd is not None:
                try:
                    import comfy.sd
                    self.vae_obj = comfy.sd.VAE(sd=sd, device=self.device)
                    self.backend = "comfy"
                    logger.info("[VAE Loader] Initialized via ComfyUI core.")
                    return
                except Exception as e:
                    logger.warning(f"[VAE Loader] ComfyUI core failed: {e}. Trying diffusers fallback...")

            # diffusers fallback — try Wan-specific VAE first, then generic AutoencoderKL
            try:
                from diffusers import AutoencoderKL, AutoModel
                
                path_dir = os.path.dirname(path) if path else ""
                
                if m_type in ("wan", "wan2.1", "wanvideo"):
                    try:
                        from diffusers import AutoencoderKLWan
                        self.vae_obj = AutoencoderKLWan.from_pretrained(
                            "Wan-AI/Wan2.1-T2V-14B", subfolder="vae", 
                            torch_dtype=torch.bfloat16
                        ).to(dev)
                        self.backend = "diffusers_wan"
                    except (ImportError, Exception):
                        self.vae_obj = AutoModel.from_pretrained(
                            path_dir if (path_dir and os.path.isdir(path_dir)) else "Wan-AI/Wan2.1-T2V-14B", torch_dtype=torch.bfloat16
                        ).to(dev)
                        self.backend = "diffusers_auto"
                elif m_type in ("ltx-video", "ltx", "ltxav"):
                    try:
                        from diffusers import AutoencoderKLLTXVideo
                        self.vae_obj = AutoencoderKLLTXVideo.from_pretrained(
                            "Lightricks/LTX-Video", subfolder="vae",
                            torch_dtype=torch.bfloat16
                        ).to(dev)
                        self.backend = "diffusers_ltx"
                    except (ImportError, Exception):
                        self.vae_obj = AutoModel.from_pretrained(
                            path_dir if (path_dir and os.path.isdir(path_dir)) else "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
                        ).to(dev)
                        self.backend = "diffusers_auto"
                elif m_type in ("hunyuanvideo", "hunyuan", "hyvideo"):
                    try:
                        from diffusers import AutoencoderKLHunyuanVideo
                        self.vae_obj = AutoencoderKLHunyuanVideo.from_pretrained(
                            "hunyuanvideo/HunyuanVideo", subfolder="vae",
                            torch_dtype=torch.bfloat16
                        ).to(dev)
                        self.backend = "diffusers_hunyuan"
                    except (ImportError, Exception):
                        self.vae_obj = AutoencoderKL.from_pretrained(
                            path_dir if (path_dir and os.path.isdir(path_dir)) else "hunyuanvideo/HunyuanVideo", torch_dtype=torch.bfloat16
                        ).to(dev)
                        self.backend = "diffusers_kl"
                else:
                    # Generic KL-VAE (Flux, SD3, SDXL, SD1.5, etc.)
                    self.vae_obj = AutoencoderKL.from_pretrained(
                        path_dir if (path_dir and os.path.isdir(path_dir)) else path, torch_dtype=torch.float16
                    ).to(dev)
                    self.backend = "diffusers_kl"
                
                logger.info(f"[VAE Loader] Initialized via diffusers ({self.backend}).")
            except Exception as e:
                logger.error(f"[VAE Loader] All VAE backends failed: {e}")
                raise RuntimeError("Could not load VAE via ComfyUI or diffusers.")

        def encode(self, pixels: torch.Tensor):
            """pixels: (B, H, W, 3) in [0, 1]"""
            if self.backend == "comfy":
                return self.vae_obj.encode(pixels)
            
            # diffusers expectations: (B, C, H, W) and often normalized to [-1, 1]
            t = pixels.permute(0, 3, 1, 2).to(self.device)
            if "wan" not in self.backend and "ltx" not in self.backend:
                t = t * 2.0 - 1.0  # [0,1] -> [-1,1]
            
            # 3D Video VAEs expect 5D input: (B, C, T, H, W)
            if "wan" in self.backend or "ltx" in self.backend or "hunyuan" in self.backend:
                t = t.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
            
            # Match VAE weight data type (e.g., torch.bfloat16)
            try:
                model_dtype = next(self.vae_obj.parameters()).dtype
                t = t.to(model_dtype)
            except Exception:
                pass
            
            with torch.no_grad():
                res = self.vae_obj.encode(t)
                latent = getattr(res, "latent_dist", res)
                if hasattr(latent, "sample"):
                    return latent.sample()
                return latent

    return StandaloneVAE(state_dict, model_type, device)

# ─────────────────────────────────────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_hdr_dataloader(
    pair_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512),
    augment: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience factory for the offline HDR DataLoader.

    Args:
        pair_dir:    Directory with pre-generated .npz pairs.
        batch_size:  Training batch size.
        num_workers: DataLoader workers.
        image_size:  (W, H) — must match the size used during pair generation.
        augment:     Enable horizontal flip augmentation.
        pin_memory:  Faster GPU transfer (disable if OOM).

    Returns a DataLoader yielding (latent_batch, log_coded_batch) tuples.
    """
    dataset = HDRPairDataset(
        pair_dir=pair_dir,
        image_size=image_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate Radiance HDR training pairs (Standalone CLI)")
    parser.add_argument("--exr_dir", required=True, nargs="+", help="Directories containing EXR/HDR files")
    parser.add_argument("--output_dir", required=True, help="Directory to save .npz pairs")
    parser.add_argument("--vae_path", required=True, help="Path to teacher VAE checkpoint (ae/safetensors)")
    try:
        from config.model_map import MODEL_VAE_CONFIG as _CFG
        _vae_choices = list(_CFG.keys())
    except ImportError:
        _vae_choices = ["flux", "sdxl", "wan", "sd3", "hunyuanvideo", "ltx-video", "sd15", "cogvideox", "lumina2", "pixart", "kolors", "aura_flow"]
    parser.add_argument("--vae_type", default="flux", choices=_vae_choices,
                        help="VAE architecture — determines latent channels and scale factor")
    parser.add_argument("--size", type=int, default=512, help="Image size (square)")
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit number of source images to process")
    parser.add_argument("--log_curve", default=None,
                        help="Log curve for targets (default: auto-select from model config)")
    parser.add_argument("--crops_per_image", type=int, default=1, help="Number of random crops per image (augmentation)")
    parser.add_argument("--target_count", type=int, default=-1, help="Target total number of pairs (auto-calculates crops_per_image)")
    parser.add_argument("--device", default="cuda", help="Computation device (cuda/cpu)")
    
    args = parser.parse_args()

    # Resolve log_curve from unified config if not specified
    if args.log_curve is None:
        try:
            from config.model_map import resolve_model_vae_config
            cfg = resolve_model_vae_config(args.vae_type)
            args.log_curve = cfg.get("log_curve", "ARRI LogC4") if cfg else "ARRI LogC4"
        except ImportError:
            args.log_curve = "ARRI LogC4"
        logger.info(f"[HDR Dataset] Auto-selected log_curve={args.log_curve} for model_type={args.vae_type}")
    
    # 1. Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 2. Load Teacher VAE
    try:
        vae = load_vae_standalone(args.vae_path, args.vae_type, args.device)
    except Exception as e:
        logger.error(f"Failed to load VAE: {e}")
        sys.exit(1)
        
    # 3. Generate Pairs
    logger.info("Starting dataset generation...")
    n_pairs = HDRPairDataset.generate_pairs(
        exr_dirs=args.exr_dir,
        output_dir=args.output_dir,
        vae=vae,
        image_size=(args.size, args.size),
        log_curve=args.log_curve,
        max_samples=args.max_samples,
        device=args.device,
        crops_per_image=args.crops_per_image,
        target_count=args.target_count
    )
    
    logger.info(f"Done! Successfully generated {n_pairs} pairs in {args.output_dir}")
