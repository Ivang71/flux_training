import os, asyncio, base64, shutil, random, string, logging, sys, requests, glob, cv2, re
import insightface, torch, pickle, time, bencodepy, hashlib,  json, subprocess, multiprocessing
from sklearn.metrics.pairwise import euclidean_distances
from math import ceil
from sklearn.cluster import DBSCAN, KMeans
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel, AutoFeatureExtractor




face_analysis = insightface.app.FaceAnalysis()
face_analysis.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.78)

device = "cuda"
gino_model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(gino_model_id)
g_model = AutoModelForZeroShotObjectDetection.from_pretrained(gino_model_id).to(device)
vision_model_id = "OpenGVLab/InternViT-300M-448px-V2_5"
vision_model = AutoModel.from_pretrained(vision_model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
vision_extractor = AutoFeatureExtractor.from_pretrained(vision_model_id, trust_remote_code=True)

# cluster by character

# ----------------------------------------
# 1) Face detection & embedding setup
# ----------------------------------------

def get_face(image_path):
    """
    Returns metadata for the best face in the image, or None if no face.
    Best face is the one with the lowest sum of pairwise cos sims among all faces in the frame (the most unique).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    faces = face_analysis.get(img)
    if len(faces) == 0:
        return None

    if len(faces) == 1:
        return extract_metadata(faces[0], image_path)

    embeddings = []
    for f in faces:
        # face.normed_embedding is a 512-D vector
        emb = f.normed_embedding.reshape(1, -1)  # shape (1, 512)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)  # shape (num_faces, 512)
    sums = cosine_similarity(embeddings).sum(axis=1)
    idx_min = np.argmin(sums)
    return extract_metadata(faces[idx_min], image_path)

def extract_metadata(face_obj, image_path):
    return {
        "image_path": image_path,
        "face_bbox": [face_obj.bbox.astype(int)],
        "face_embedding": face_obj.normed_embedding.tolist(), # shape (512,)
    }

# ----------------------------------------
# 2) Collect metadata from frames
# ----------------------------------------
def get_frames(frames_folder):
    metadata = []
    frame_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
    for path in frame_paths:
        face_data = get_face(path)
        if face_data is None:
            os.remove(path)
        else:
            metadata.append(face_data)
    return metadata


# ----------------------------------------
# 3) Clustering by character
# ----------------------------------------
def cluster_by_char(
    face_metadata,
    eps=0.8,
    min_samples=10,
    max_cluster_size=10000,
    merge_centroid_dist=0.3
):
    """
    1) DBSCAN to form initial clusters.
    2) Merge clusters whose centroids are within 'merge_centroid_dist'.
    3) Split large clusters using K-means if > max_cluster_size.
    
    face_metadata: list of dict, each with 'face_embedding': list[float], 'image_path', ...
    eps: DBSCAN eps (bigger => merges more frames into same cluster).
    min_samples: DBSCAN min_samples (small => easier to form a cluster).
    max_cluster_size: after clustering, if a cluster has > max_cluster_size frames, we split it.
    merge_centroid_dist: if centroids of two clusters are closer than this, merge them.
    """
    embeddings = np.array([item["face_embedding"] for item in face_metadata])

    # Step 1: DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(embeddings)

    # Collect cluster members
    clusters_dict = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue  # noise
        clusters_dict.setdefault(label, []).append(idx)

    # Convert to a list of clusters
    clusters = list(clusters_dict.values())

    if not clusters:
        return []

    # Step 2: Merge cluster centroids if they are too close
    #   - Compute centroid of each cluster
    centroids = []
    for c in clusters:
        emb_c = embeddings[c]
        centroid = emb_c.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)  # shape (num_clusters, emb_dim)

    #   - Compute distance matrix between centroids
    dist_mat = euclidean_distances(centroids, centroids)
    #   - Merge clusters if distance < merge_centroid_dist
    #     We'll do a simple union-find or BFS approach
    visited = [False]*len(clusters)
    merged_clusters = []

    def dfs(idx, group):
        stack = [idx]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            group.append(node)
            # check neighbors
            for nbr in range(len(clusters)):
                if not visited[nbr] and dist_mat[node, nbr] < merge_centroid_dist:
                    stack.append(nbr)

    for i in range(len(clusters)):
        if not visited[i]:
            group = []
            dfs(i, group)
            merged_clusters.append(group)

    # merged_clusters is now a list of lists of cluster indices to merge
    final_merged = []
    for group in merged_clusters:
        # union of all frames from those clusters
        merged_frames = []
        for ci in group:
            merged_frames.extend(clusters[ci])
        final_merged.append(merged_frames)

    # Step 3: For each merged cluster, if it's > max_cluster_size, split via K-means
    final_clusters = []
    for frames in final_merged:
        if len(frames) <= max_cluster_size:
            final_clusters.append(frames)
        else:
            # sub-cluster with K-means
            n_sub = ceil(len(frames)/max_cluster_size)
            sub_embeddings = embeddings[frames]
            km = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
            sub_labels = km.fit_predict(sub_embeddings)

            for sub_label in range(n_sub):
                sub_indices = [frames[i] for i, sl in enumerate(sub_labels) if sl == sub_label]
                # optional: discard if < min_samples
                if len(sub_indices) >= min_samples:
                    final_clusters.append(sub_indices)
                # else: either discard or merge with nearest sub-cluster (not shown)

    return final_clusters




# adding body bounding box and embedding
def compute_body_embedding(pil_image):
    inputs = vision_extractor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(torch.bfloat16).to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = vision_model(**inputs)
    if hasattr(outputs, "pooler_output"):
        emb = outputs.pooler_output.squeeze(0)
    else:
        emb = outputs.last_hidden_state[:, 0, :]
    return emb.cpu().tolist()


def add_body_data(meta):
    im = Image.open(meta["image_path"]).convert("RGB")
    w, h = im.size
    face = meta["face_bbox"][0]
    inp = processor(images=im, text=[["a person"]], return_tensors="pt").to(device)
    with torch.no_grad():
        out = g_model(**inp)
    r = processor.post_process_grounded_object_detection(out, inp.input_ids, 0.4, 0.3, [(h, w)])[0]

    def overlap(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        A1 = (a[2] - a[0]) * (a[3] - a[1])
        A2 = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (A1 + A2 - inter) if (A1 + A2 - inter) else 0

    best_box, best_o = None, 0
    for box, sc, lab in zip(r["boxes"], r["scores"], r["labels"]):
        box = box.tolist()
        o = overlap(face, box)
        if o > best_o:
            best_o = o
            best_box = box

    if not best_box:
        return None

    bx1, by1, bx2, by2 = map(int, best_box)
    cropped_body = im.crop((bx1, by1, bx2, by2))
    meta["body_bbox"] = best_box
    meta["body_embedding"] = compute_body_embedding(cropped_body)
    return meta
