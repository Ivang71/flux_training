import os, asyncio, base64, shutil, random, string, logging, sys, requests, glob, cv2, re
import insightface, torch, pickle, time, bencodepy, hashlib,  json, subprocess, multiprocessing


def upload_to_drive(stage=0):
    DATASET_DIR = "data/dataset"
    REMOTE_BASE = f"drive:dataset_stage_{stage}"
    bundles = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.isdigit()]
    for bundle in bundles:
        archive_name = f"{bundle}.tar"  # uncompressed archive
        subprocess.run(["tar", "-cf", archive_name, "-C", DATASET_DIR, bundle], check=True)
        subprocess.run([
            "rclone", "copy", archive_name, REMOTE_BASE,
            "--checksum", "--transfers=32", "--checkers=32", "--fast-list", "--progress"
        ], check=True)
        os.remove(archive_name)
        logging.info(f"Processed and uploaded bundle {bundle}")
