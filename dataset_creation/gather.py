import os, asyncio, base64, shutil, random, string, logging, sys, requests, glob, cv2, re, uvloop
import insightface, torch, pickle, time, bencodepy, hashlib,  json, subprocess, multiprocessing
from huggingface_hub import login
from torrentp import TorrentDownloader
from tqdm import tqdm
from os.path import join, isdir
from copy import deepcopy
import warnings
from utils.utils import upload_to_drive
from utils.face_body_detection import get_frames, cluster_by_char, add_body_data
# import nest_asyncio
# nest_asyncio.apply()

# asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

warnings.simplefilter("ignore", FutureWarning)

k = base64.b64decode('aGZfaHZqck9VTXFvTXF3dW9HR3JoTlZKSWlsZUtFTlNQbXRjTw==').decode()
login(token=k, add_to_git_credential=False)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = k
os.environ['HF_TOKEN'] = k
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    filename="gather.log", filemode="w")



async def upload_bundle(bundle, local, remote):
    archive = f"{bundle}.tar"
    proc = await asyncio.create_subprocess_exec(
        "tar", "-cf", archive, "-C", local, bundle,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    proc = await asyncio.create_subprocess_exec(
        "rclone", "copy", archive, remote,
        "--checksum", "--transfers=64", "--checkers=64",
        "--fast-list", "--multi-thread-streams=4", "--progress",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    os.remove(archive)
    logging.info(f"Processed and uploaded bundle {bundle}")


async def upload_to_drive(stage=0):
    local = "data/dataset"
    remote = f"drive:dataset/stage_{stage}"
    bundles = [d for d in os.listdir(local) if os.path.isdir(os.path.join(local, d)) and d.isdigit()]
    if bundles:
        await asyncio.gather(*(upload_bundle(bundle, local, remote) for bundle in bundles))


async def sync_loop():
    while True:
        await upload_to_drive(0)
        await asyncio.sleep(12*60)



# download & extract frames
def randStr(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def get_url(keyword):
    try:
        params = {"query_term": keyword, "sort_by": "peers", "order_by": "desc", "limit": 50}
        r = requests.get("https://yts.mx/api/v2/list_movies.json", params=params)
        r.raise_for_status()
        movies = r.json().get('data', {}).get('movies', [])
        torrents = (
            torrent for movie in movies for torrent in movie["torrents"]
            if torrent.get("quality") in ("1080p", "720p")
        )
        best = max(torrents, key=lambda t: t["peers"])
        return best['url']
    except Exception as e:
        logging.exception(f"Error: {e}")
        return None

def download_sync(movie_name, save_path):
    os.makedirs('temp', exist_ok=True)
    torrent_file = f"temp/{re.sub(r'[^a-zA-Z0-9_.-]', '_', movie_name)}.torrent"
    url = get_url(movie_name)
    with open(torrent_file, "wb") as f:
        f.write(requests.get(url).content)
    torrent = TorrentDownloader(torrent_file, save_path)
    asyncio.run(torrent.start_download())
    os.remove(torrent_file)

async def download(movie_name, save_path, timeout=360):
    proc = multiprocessing.Process(target=download_sync, args=(movie_name, save_path))
    proc.start()
    try:
        await asyncio.wait_for(asyncio.to_thread(proc.join), timeout=timeout)
    except asyncio.TimeoutError:
        logging.info(f"Timeout of {timeout}s reached; aborting download.")
        proc.terminate()
        proc.join()



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


def get_video_duration(v):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", v]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return float(json.loads(out)['format']['duration'])

async def extract_segment(name, v, s, d, idx):
    od = f"./data/raw_frames/{name}/segment_{idx}"
    os.makedirs(od, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-ss", str(s), "-t", str(d),
           "-i", v, "-vf", "hwdownload,format=nv12,fps=0.5", "-q:v", "2", "-vsync", "0", "-threads", "0", f"{od}/frame_%04d.jpg"]
    proc = await asyncio.create_subprocess_exec(*cmd)
    await proc.wait()

def merge_frames(name, parts):
    bd = f"./data/raw_frames/{name}"
    cnt = 1
    for i in range(1, parts + 1):
        seg = os.path.join(bd, f"segment_{i}")
        for f in sorted(glob.glob(os.path.join(seg, '*.jpg'))):
            shutil.move(f, os.path.join(bd, f"frame_{cnt:04d}.jpg"))
            cnt += 1
        os.rmdir(seg)

async def extract_frames(name, v, parts=5):
    bd = f"./data/raw_frames/{name}"
    os.makedirs(bd, exist_ok=True)
    total = get_video_duration(v)
    d = total / parts
    tasks = [asyncio.create_task(extract_segment(name, v, i * d, d, i + 1)) for i in range(parts)]
    await asyncio.gather(*tasks)
    merge_frames(name, parts)


async def download_extract_frames(movie_name, name, vids_folder):
    save_path = f'./data/{name}'
    t = time.time()
    await download(movie_name, save_path)
    dTime = round(time.time() - t, 4)
    logging.info(f'Downloading took {dTime} seconds')
    
    ext = extract_video(save_path, vids_folder, name)
    vid_path = f"./data/vids/{name + ext}"
    t = time.time()
    await extract_frames(name, vid_path)
    eTime = round(time.time() - t, 4)
    logging.info(f'Frame extraction took {eTime} seconds')
    shutil.rmtree(save_path, ignore_errors=True)
    os.remove(vid_path)
    return dTime, eTime


# add to dataset
def get_max_folder_number(directory):
    return max((int(d) for d in os.listdir(directory) if d.isdigit() and isdir(join(directory, d))), default=0)



async def process_movie(movie_name):
    name = randStr()
    os.makedirs('data', exist_ok=True)
    vids_folder = "./data/vids"

    try:
        # Wait up to 12 minutes for the processing to finish.
        dTime, eTime = await asyncio.wait_for(
            download_extract_frames(movie_name, name, vids_folder), timeout=12*60
        )
    except asyncio.TimeoutError:
        return "download_extract_frames timeout"
    t = time.time()
    frames_folder = f"./data/raw_frames/{name}"
    logging.info(f"Collecting face entries...")
    metadata = await asyncio.to_thread(get_frames, frames_folder) # extracting frames with faces
    fTime = round(time.time() - t, 4)
    logging.info(f"Collected {len(metadata)} face entries.")
    logging.info(f'Filtering out frames without faces took {fTime} seconds')

    t = time.time()
    raw_clusters = await asyncio.to_thread(cluster_by_char, metadata) # clustering by character
    min_size, max_size = 5, 11
    filtered_clusters = [cluster for cluster in raw_clusters if len(cluster) >= min_size]
    clusters = [random.sample(cluster, min(len(cluster), max_size)) for cluster in filtered_clusters]
    for i, c in enumerate(clusters):
        logging.info(f"Cluster {i} has {len(c)} frames.")
    cTime = round(time.time() - t, 4)
    logging.info(f'Clustering took {cTime} seconds')

    t = time.time()
    for i, cluster in enumerate(clusters): # adding body bounding box and body embedding
        for idx in tqdm(cluster, desc=f"Cluster {i}"):
            result = await asyncio.to_thread(add_body_data, metadata[idx])
            if result is None:
                metadata[idx]["defective"] = True
            else:
                metadata[idx] = result
    bTime = round(time.time() - t, 4)
    logging.info(f'Processing bodies took {bTime} seconds')

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
            new_folder = join(base, str(bundle))
            os.makedirs(new_folder, exist_ok=True)
            with open(join(new_folder, "info.json"), "w") as f:
                json.dump({"all_labeled": False, "labeled": []}, f)

        char_dir = join(base, str(bundle), str(char))
        os.makedirs(char_dir, exist_ok=True)

        char_data = []
        img_index = 0

        for mtd_i in cluster:
            if metadata[mtd_i].get('defective', False):
                continue

            src_path = metadata[mtd_i]['image_path']
            dest_image_name = f'{img_index}.jpg'
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
    logging.info(f"Downloading took {dTime} seconds")
    logging.info(f"Extraction took {eTime} seconds")
    logging.info(f"Filtering took {fTime} seconds")
    logging.info(f"Clustering took {cTime} seconds")
    logging.info(f"Body processing took {bTime} seconds")
    logging.info(f'Adding to dataset took {aTime} seconds')

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
    movies = [movie for movie in movies if movie not in processed]
    
    upload_tasks = []
    
    while movies:
        movie = movies.pop(0)
        try:
            result = await process_movie(movie)
        except Exception as e:
            logging.exception(f"Exception processing {movie}: {e}")
            result = "something went wrong while processing"

        if result == 0:
            logging.info(f"Successfully processed {movie}")
            processed.append(movie)
        else:
            logging.exception(f"Processing error ({result}) for {movie}")
        
        with open(movies_path, "w") as f:
            json.dump(movies, f, indent=4)
        with open(processed_path, "w") as f:
            json.dump(processed, f, indent=4)
        
        # Launch the upload function in the background asynchronously.
        # This runs upload_to_drive(0) in a separate thread.
        task = asyncio.create_task(asyncio.to_thread(upload_to_drive, 0))
        upload_tasks.append(task)
        
        # Optionally yield control so upload can start concurrently.
        await asyncio.sleep(0)
    
    # Wait for all upload tasks to finish before exiting.
    await asyncio.gather(*upload_tasks)


if __name__ == '__main__':
    async def runner():
        sync_task = asyncio.create_task(sync_loop())
        result = await main()
        sync_task.cancel()
        return result

    sys.exit(asyncio.run(runner()))
