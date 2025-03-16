import os, asyncio, base64, shutil, random, string, argparse, sys, requests, glob, cv2
import insightface, torch, pickle, time, bencodepy, hashlib,  json, subprocess
from huggingface_hub import login
from torrentp import TorrentDownloader
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from math import ceil
from sklearn.cluster import DBSCAN, KMeans
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel, AutoFeatureExtractor
from os.path import join, isdir
from copy import deepcopy
import nest_asyncio
nest_asyncio.apply()

k = base64.b64decode('aGZfaHZqck9VTXFvTXF3dW9HR3JoTlZKSWlsZUtFTlNQbXRjTw==').decode()
login(token=k, add_to_git_credential=False)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = k
os.environ['HF_TOKEN'] = k
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"


def torrent_url_to_magnet(torrent_url):
    r = requests.get(torrent_url)
    r.raise_for_status()
    torrent_dict = bencodepy.decode(r.content)
    info = torrent_dict[b'info']
    info_encoded = bencodepy.encode(info)
    info_hash = hashlib.sha1(info_encoded).hexdigest()
    return f"magnet:?xt=urn:btih:{info_hash}".strip()

def get_magnet(keyword):
    r = requests.get("https://yts.mx/api/v2/list_movies.json", params={"query_term": keyword})
    r.raise_for_status()
    movies = r.json().get('data', {}).get('movies', [])
    fallback = None
    for movie in reversed(movies):
        torrents = movie.get('torrents', [])
        if torrents and fallback is None:
            fallback = torrent_url_to_magnet(torrents[0].get('url'))
        for t in torrents:
            if t.get('quality') == '1080p':
                return torrent_url_to_magnet(t.get('url'))
    return fallback


# download & extract frames
def randStr(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


async def download(save_path, magnet_uri):
    torrent_file = TorrentDownloader(magnet_uri, save_path=save_path)
    await torrent_file.start_download()


def extract_video(source_folder, target_folder, name):
    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")
    os.makedirs(target_folder, exist_ok=True)

    for root, _, files in os.walk(source_folder):
        for filename in files:
            if filename.lower().endswith(video_extensions):
                source_path = os.path.join(root, filename)
                _, ext = os.path.splitext(filename)
                destination_path = os.path.join(target_folder, name+ext)
                shutil.move(source_path, destination_path)
                return ext # Stop after moving the first found video file


async def extract_frames(name, vid_path):
    mkdir_cmd = ["mkdir", "-p", f"./data/raw_frames/{name}"]
    proc_mkdir = await asyncio.create_subprocess_exec(*mkdir_cmd)
    await proc_mkdir.wait()

    ffmpeg_cmd = [
        "ffmpeg",
        "-hwaccel", "cuda",
        "-c:v", "h264_cuvid",
        "-i", vid_path,
        "-vf", "select='not(mod(n,50))'",
        "-q:v", "2",
        "-vsync", "0",
        "-threads", "0",
        f"./data/raw_frames/{name}/frame_%04d.jpg"
    ]
    proc_ffmpeg = await asyncio.create_subprocess_exec(*ffmpeg_cmd)
    await proc_ffmpeg.wait()


async def download_extract_frames(magnet_uri, name, vids_folder):
    save_path = f'./data/{name}'
    t = time.time()
    await download(save_path, magnet_uri)
    dTime = round(time.time() - t, 4)
    print(f'Downloading took {dTime} seconds')
    
    ext = extract_video(save_path, vids_folder, name)
    vid_path = f"./data/vids/{name + ext}"
    t = time.time()
    await extract_frames(name, vid_path)
    eTime = round(time.time() - t, 4)
    print(f'Frame extraction {eTime} seconds')
    shutil.rmtree(save_path, ignore_errors=True)
    os.remove(vid_path)
    return dTime, eTime


# cluster by character

# ----------------------------------------
# 1) Face detection & embedding setup
# ----------------------------------------

def get_face(image_path, face_analysis):
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
    face_analysis = insightface.app.FaceAnalysis()
    face_analysis.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.78)

    metadata = []
    frame_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
    for path in frame_paths:
        face_data = get_face(path, face_analysis)
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
device = "cuda"
gino_model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(gino_model_id)
g_model = AutoModelForZeroShotObjectDetection.from_pretrained(gino_model_id).to(device)
vision_model_id = "OpenGVLab/InternViT-300M-448px-V2_5"
vision_model = AutoModel.from_pretrained(vision_model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
vision_extractor = AutoFeatureExtractor.from_pretrained(vision_model_id, trust_remote_code=True)

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

def detect_body_bbox_and_embedding(meta):
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


# add to dataset
def get_max_folder_number(directory):
    return max((int(d) for d in os.listdir(directory) if d.isdigit() and isdir(join(directory, d))), default=0)


def upload_to_drive():
    DATASET_DIR = "data/dataset"
    REMOTE_BASE = "drive:dataset"
    bundles = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.isdigit()]
    for bundle in bundles:
        archive_name = f"{bundle}.tar"  # uncompressed archive
        subprocess.run(["tar", "-cf", archive_name, "-C", DATASET_DIR, bundle], check=True)
        subprocess.run([
            "rclone", "copy", archive_name, REMOTE_BASE,
            "--transfers=32", "--checkers=32", "--fast-list", "--progress"
        ], check=True)
        os.remove(archive_name)
        print(f"Processed and uploaded bundle {bundle}")








def process_movie(movie_name):
    name = randStr()
    name = "bopxYeeJ"
    os.makedirs('data', exist_ok=True)
    vids_folder = "./data/vids"

    magnet = get_magnet(movie_name) # downloading
    dTime, eTime = asyncio.run(download_extract_frames(magnet, name, vids_folder))

    t = time.time()
    frames_folder = f"./data/raw_frames/{name}"
    metadata = get_frames(frames_folder) # extracting frames with faces
    fTime = round(time.time() - t, 4)
    print(f"Collected {len(metadata)} face entries.")
    print(f'Filtering out frames without faces took {fTime} seconds')

    t = time.time()
    raw_clusters = cluster_by_char(metadata) # clustering by character
    min_size, max_size = 13, 36
    filtered_clusters = [cluster for cluster in raw_clusters if len(cluster) >= min_size]
    clusters = [random.sample(cluster, min(len(cluster), max_size)) for cluster in filtered_clusters]
    for i, c in enumerate(clusters):
        print(f"Cluster {i} has {len(c)} frames.")
    cTime = round(time.time() - t, 4)
    print(f'Clustering took {cTime} seconds')

    
    t = time.time()
    for i, cluster in enumerate(clusters): # adding body bounding box and body embedding
        for idx in tqdm(cluster, desc=f"Cluster {i}"):
            result = detect_body_bbox_and_embedding(metadata[idx])
            if result is None:
                metadata[idx]["defective"] = True
            else:
                metadata[idx] = result
    bTime = round(time.time() - t, 4)
    print(f'Processing bodies took {bTime} seconds')

    t = time.time()
    base = join(os.getcwd(), 'data', 'dataset') # add to dataset
    os.makedirs(base, exist_ok=True)
    bundle = get_max_folder_number(base)
    os.makedirs(join(base, str(bundle)), exist_ok=True)
    char = get_max_folder_number(join(base, str(bundle)))

    for cluster in clusters:
        if char > 999:
            bundle += 1
            char = 0
            os.makedirs(join(base, str(bundle)), exist_ok=True)

        char_dir = join(base, str(bundle), str(char))
        os.makedirs(char_dir, exist_ok=True)

        char_data = []
        img_index = 0

        for mtd_i in cluster:
            if metadata[mtd_i].get('defective', False):
                continue

            src_path = metadata[mtd_i]['image_path']
            dest_image_name = f'char_data_{img_index}.jpg'
            dest_path = join(char_dir, dest_image_name)

            shutil.copy2(src_path, dest_path)

            frame_copy = deepcopy(metadata[mtd_i])
            frame_copy['image_path'] = dest_path
            char_data.append(frame_copy)
            img_index += 1

        with open(join(char_dir, 'char_data.pkl'), 'wb') as f:
            pickle.dump(char_data, f)
        char += 1
    aTime = round(time.time() - t, 4)

    shutil.rmtree(frames_folder, ignore_errors=True)
    print(f"Downloading took {dTime} seconds")
    print(f"Extraction took {eTime} seconds")
    print(f"Filtering took {fTime} seconds")
    print(f"Clustering took {cTime} seconds")
    print(f"Body processing took {bTime} seconds")
    print(f'Adding to dataset took {aTime} seconds')

    return 0


async def main():
    movies_path = "./movies.json"
    processed_path = "./processed.json"
    
    with open(movies_path, "r") as f:
        movies = json.load(f)
    if os.path.exists(processed_path):
        with open(processed_path, "r") as f:
            processed = json.load(f)
    else:
        processed = []
    
    upload_tasks = []
    
    while movies:
        movie = movies.pop(0)
        result = process_movie(movie)
        if result.returncode == 0:
            print(f"Successfully processed {movie}")
            processed.append(movie)
        else:
            print(f"Processing error ({result.returncode}) for {movie}")
        
        with open(movies_path, "w") as f:
            json.dump(movies, f, indent=4)
        with open(processed_path, "w") as f:
            json.dump(processed, f, indent=4)
        
        # Launch the upload function in the background asynchronously.
        # This runs upload_to_drive() in a separate thread.
        task = asyncio.create_task(asyncio.to_thread(upload_to_drive))
        upload_tasks.append(task)
        
        # Optionally yield control so upload can start concurrently.
        await asyncio.sleep(0)
    
    # Wait for all upload tasks to finish before exiting.
    await asyncio.gather(*upload_tasks)


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
