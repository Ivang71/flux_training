import os, asyncio, base64, shutil, random, string, logging, sys, requests, glob, cv2, re
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

warnings.simplefilter("ignore", FutureWarning)

k = base64.b64decode('aGZfaHZqck9VTXFvTXF3dW9HR3JoTlZKSWlsZUtFTlNQbXRjTw==').decode() # never change this token
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


class RateLimiter:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self.last_call = 0

    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call
        if time_since_last < self.min_interval:
            time.sleep(self.min_interval - time_since_last)
        self.last_call = time.time()

# Create a global rate limiter for YTS API calls
yts_limiter = RateLimiter(3.0)  # 3 seconds between calls

def get_url(keyword):
    try:
        yts_limiter.wait()  # Wait if needed to respect rate limit
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
    if url is None:
        logging.error(f"Failed to get torrent URL for {movie_name}")
        return
    try:
        with open(torrent_file, "wb") as f:
            f.write(requests.get(url).content)
        torrent = TorrentDownloader(torrent_file, save_path)
        asyncio.run(torrent.start_download())
        
        # Wait for files to be fully written
        time.sleep(2)
        
        # Verify download location
        if not os.path.exists(save_path):
            logging.error(f"Download folder {save_path} was not created")
            return
            
        # Check if any video files exist
        video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")
        has_video = False
        for root, _, files in os.walk(save_path):
            for filename in files:
                if filename.lower().endswith(video_extensions):
                    has_video = True
                    logging.info(f"Found video file: {os.path.join(root, filename)}")
                    break
            if has_video:
                break
                
        if not has_video:
            logging.error(f"No video files found in {save_path} or its subdirectories")
    except Exception as e:
        logging.error(f"Error downloading {movie_name}: {e}")
    finally:
        if os.path.exists(torrent_file):
            os.remove(torrent_file)

async def download(movie_name, save_path, timeout=540):
    proc = multiprocessing.Process(target=download_sync, args=(movie_name, save_path))
    proc.start()
    try:
        await asyncio.wait_for(asyncio.to_thread(proc.join), timeout=timeout)
        # Verify download completed successfully
        if not os.path.exists(save_path):
            logging.error(f"Download folder {save_path} was not created")
            return False
        # Check if any video files exist
        video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")
        has_video = any(
            filename.lower().endswith(video_extensions)
            for root, _, files in os.walk(save_path)
            for filename in files
        )
        if not has_video:
            logging.error(f"No video files found in {save_path}")
            return False
        return True
    except asyncio.TimeoutError:
        logging.error(f"Timeout of {timeout}s reached; aborting download.")
        proc.terminate()
        proc.join()
        return False
    except Exception as e:
        logging.error(f"Error during download: {e}")
        return False



def extract_video(source_folder, target_folder, name):
    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")
    os.makedirs(target_folder, exist_ok=True)

    # Wait for download to complete and verify folder exists
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        if not os.path.exists(source_folder):
            logging.warning(f"Source folder {source_folder} not found, attempt {attempt + 1}/{max_retries}")
            time.sleep(retry_delay)
            continue

        # List all files in source folder and its subdirectories
        video_files = []
        for root, _, files in os.walk(source_folder):
            for filename in files:
                if filename.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, filename))
        
        if not video_files:
            logging.warning(f"No video files found in {source_folder} or its subdirectories, attempt {attempt + 1}/{max_retries}")
            time.sleep(retry_delay)
            continue

        # Try each video file until we successfully move one
        for source_path in video_files:
            try:
                if not os.path.exists(source_path):
                    logging.warning(f"Video file {source_path} not found, skipping")
                    continue
                    
                _, ext = os.path.splitext(source_path)
                destination_path = os.path.join(target_folder, name + ext)
                
                # Ensure the file is not being written to
                if os.path.getsize(source_path) == 0:
                    logging.warning(f"Video file {source_path} is empty, skipping")
                    continue
                
                shutil.move(source_path, destination_path)
                if not os.path.exists(destination_path):
                    logging.error(f"Failed to move file to {destination_path}")
                    continue
                    
                logging.info(f"Successfully moved video file from {source_path} to {destination_path}")
                return ext
            except Exception as e:
                logging.error(f"Error moving file {source_path}: {e}")
                continue

    logging.error(f"Failed to find or move video file after {max_retries} attempts")
    return None


def get_video_duration(v):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", v]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return float(json.loads(out)['format']['duration'])

async def extract_segment(name, v, s, d, idx):
    od = f"./data/raw_frames/{name}/segment_{idx}"
    os.makedirs(od, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-hwaccel", "nvdec", "-hwaccel_output_format", "cuda", "-ss", str(s), "-t", str(d),
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


def cleanup_data_folders():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        return
        
    allowed_folders = {"dataset", "raw_frames", "vids"}
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item not in allowed_folders:
            shutil.rmtree(item_path, ignore_errors=True)
            logging.info(f"Cleaned up folder: {item_path}")

def cleanup_movie_data(name):
    """Clean up all temporary files and folders for a specific movie"""
    paths_to_clean = [
        f'./data/{name}',  # Download folder
        f'./data/vids/{name}.mp4',  # Video file
        f'./data/vids/{name}.mkv',  # Video file
        f'./data/vids/{name}.avi',  # Video file
        f'./data/raw_frames/{name}',  # Frames folder
        f'temp/{name}.torrent'  # Torrent file
    ]
    
    for path in paths_to_clean:
        if os.path.isfile(path):
            try:
                os.remove(path)
                logging.info(f"Cleaned up file: {path}")
            except Exception as e:
                logging.error(f"Failed to clean up file {path}: {e}")
        elif os.path.isdir(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
                logging.info(f"Cleaned up directory: {path}")
            except Exception as e:
                logging.error(f"Failed to clean up directory {path}: {e}")
    
    # Completely remove and recreate temp folder
    try:
        if os.path.exists('temp'):
            shutil.rmtree('temp', ignore_errors=True)
            logging.info("Removed temp folder")
        os.makedirs('temp', exist_ok=True)
        logging.info("Recreated temp folder")
    except Exception as e:
        logging.error(f"Failed to clean up temp folder: {e}")

async def download_frames(movie_name, save_path):
    try:
        t = time.time()
        success = await download(movie_name, save_path)
        if not success:
            logging.error(f"Download failed for {movie_name}")
            cleanup_movie_data(os.path.basename(save_path))
            return None
            
        dTime = round(time.time() - t, 4)
        logging.info(f'Downloading took {dTime} seconds')
        return dTime
    except Exception as e:
        logging.exception(f"Error downloading {movie_name}: {e}")
        cleanup_movie_data(os.path.basename(save_path))
        return None

async def extract_frames_from_video(name, vids_folder):
    try:
        save_path = f'./data/{name}'
        
        # Verify download folder exists and contains files
        if not os.path.exists(save_path):
            logging.error(f"Download folder {save_path} does not exist")
            cleanup_movie_data(name)
            return None
            
        # List files in download folder
        files = []
        for root, _, filenames in os.walk(save_path):
            files.extend(filenames)
            
        if not files:
            logging.error(f"No files found in download folder {save_path}")
            cleanup_movie_data(name)
            return None
            
        ext = extract_video(save_path, vids_folder, name)
        if not ext:
            logging.error(f"No video file found in {save_path}")
            cleanup_movie_data(name)
            return None
            
        vid_path = f"./data/vids/{name + ext}"
        if not os.path.exists(vid_path):
            logging.error(f"Video file not found at {vid_path}")
            cleanup_movie_data(name)
            return None
            
        t = time.time()
        await extract_frames(name, vid_path)
        eTime = round(time.time() - t, 4)
        logging.info(f'Frame extraction took {eTime} seconds')
        return eTime
    except Exception as e:
        logging.exception(f"Error extracting frames for {name}: {e}")
        cleanup_movie_data(name)
        return None
    finally:
        # Clean up temporary files
        if os.path.exists(save_path):
            shutil.rmtree(save_path, ignore_errors=True)
        if 'vid_path' in locals() and os.path.exists(vid_path):
            os.remove(vid_path)


# add to dataset
def get_max_folder_number(directory):
    return max((int(d) for d in os.listdir(directory) if d.isdigit() and isdir(join(directory, d))), default=0)



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
    
    # Queue for coordinating downloads and extractions
    extraction_queue = asyncio.Queue(maxsize=5)  # Limit queue size to prevent stockpiling
    active_downloads = set()
    max_concurrent_downloads = 5
    
    async def process_extraction_queue():
        while True:
            try:
                # Get next movie to extract frames from
                movie_name, name = await extraction_queue.get()
                try:
                    # Extract frames (this runs sequentially)
                    eTime = await extract_frames_from_video(name, "./data/vids")
                    if eTime is None:
                        logging.error(f"Failed to extract frames for {movie_name}")
                        processed.append(movie_name)
                        with open(processed_path, "w") as f:
                            json.dump(processed, f, indent=4)
                        cleanup_movie_data(name)
                        continue
                    
                    # Continue with the rest of the pipeline
                    frames_folder = f"./data/raw_frames/{name}"
                    logging.info(f"Collecting face entries...")
                    t = time.time()  # Initialize time before using it
                    metadata = await asyncio.to_thread(get_frames, frames_folder)
                    fTime = round(time.time() - t, 4)
                    
                    if not metadata:
                        logging.error(f"No faces detected in frames for {movie_name}")
                        processed.append(movie_name)
                        with open(processed_path, "w") as f:
                            json.dump(processed, f, indent=4)
                        cleanup_movie_data(name)
                        continue
                        
                    logging.info(f"Collected {len(metadata)} face entries.")
                    logging.info(f'Filtering out frames without faces took {fTime} seconds')

                    t = time.time()
                    raw_clusters = await asyncio.to_thread(cluster_by_char, metadata)
                    if not raw_clusters:
                        logging.error(f"No valid clusters found for {movie_name}")
                        processed.append(movie_name)
                        with open(processed_path, "w") as f:
                            json.dump(processed, f, indent=4)
                        cleanup_movie_data(name)
                        continue
                        
                    min_size, max_size = 5, 11
                    filtered_clusters = [cluster for cluster in raw_clusters if len(cluster) >= min_size]
                    if not filtered_clusters:
                        logging.error(f"No clusters meet minimum size requirement for {movie_name}")
                        processed.append(movie_name)
                        with open(processed_path, "w") as f:
                            json.dump(processed, f, indent=4)
                        cleanup_movie_data(name)
                        continue
                        
                    clusters = [random.sample(cluster, min(len(cluster), max_size)) for cluster in filtered_clusters]
                    for i, c in enumerate(clusters):
                        logging.info(f"Cluster {i} has {len(c)} frames.")
                    cTime = round(time.time() - t, 4)
                    logging.info(f'Clustering took {cTime} seconds')

                    t = time.time()
                    for i, cluster in enumerate(clusters):
                        for idx in tqdm(cluster, desc=f"Cluster {i}"):
                            result = await asyncio.to_thread(add_body_data, metadata[idx])
                            if result is None:
                                metadata[idx]["defective"] = True
                            else:
                                metadata[idx] = result
                    bTime = round(time.time() - t, 4)
                    logging.info(f'Processing bodies took {bTime} seconds')

                    t = time.time()
                    base = join(os.getcwd(), 'data', 'dataset')
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
                    logging.info(f"Extraction took {eTime} seconds")
                    logging.info(f"Filtering took {fTime} seconds")
                    logging.info(f"Clustering took {cTime} seconds")
                    logging.info(f"Body processing took {bTime} seconds")
                    logging.info(f'Adding to dataset took {aTime} seconds')

                    processed.append(movie_name)
                    with open(processed_path, "w") as f:
                        json.dump(processed, f, indent=4)
                    
                    # Launch upload in background
                    task = asyncio.create_task(asyncio.to_thread(upload_to_drive, 0))
                    upload_tasks.append(task)
                    
                except Exception as e:
                    logging.exception(f"Exception processing {movie_name}: {e}")
                    cleanup_movie_data(name)
                finally:
                    extraction_queue.task_done()
            except asyncio.CancelledError:
                break

    # Start the extraction queue processor
    extraction_processor = asyncio.create_task(process_extraction_queue())
    
    try:
        while movies or active_downloads:
            # Only start new downloads if:
            # 1. We have movies to process
            # 2. We have capacity for new downloads
            # 3. The extraction queue is not full
            while (len(active_downloads) < max_concurrent_downloads and 
                   movies and 
                   extraction_queue.qsize() < extraction_queue.maxsize):
                movie = movies.pop(0)
                name = randStr()
                os.makedirs('data', exist_ok=True)
                
                # Create download task
                download_task = asyncio.create_task(download_frames(movie, f'./data/{name}'))
                active_downloads.add(download_task)
                
                # When download completes, add to extraction queue
                async def handle_download_completion(task, movie_name, movie_id):
                    try:
                        await task
                        # Wait for space in the extraction queue
                        await extraction_queue.put((movie_name, movie_id))
                    except Exception as e:
                        logging.exception(f"Download failed for {movie_name}: {e}")
                        cleanup_movie_data(movie_id)
                    finally:
                        active_downloads.remove(task)
                
                asyncio.create_task(handle_download_completion(download_task, movie, name))
                
                # Update movies list
                with open(movies_path, "w") as f:
                    json.dump(movies, f, indent=4)
            
            # Log queue status periodically
            if movies or active_downloads:
                logging.info(f"Queue status - Movies remaining: {len(movies)}, Active downloads: {len(active_downloads)}, Extraction queue size: {extraction_queue.qsize()}")
            
            # Wait a bit before checking again
            await asyncio.sleep(1)
    
    finally:
        # Clean up
        extraction_processor.cancel()
        await extraction_processor
        await extraction_queue.join()
        await asyncio.gather(*upload_tasks)


if __name__ == '__main__':
    async def runner():
        cleanup_data_folders()  # Clean up before starting
        sync_task = asyncio.create_task(sync_loop())
        result = await main()
        sync_task.cancel()
        return result

    sys.exit(asyncio.run(runner()))
