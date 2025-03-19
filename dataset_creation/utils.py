import os, asyncio, base64, shutil, random, string, logging, sys, requests, glob, cv2, re
import insightface, torch, pickle, time, bencodepy, hashlib,  json, subprocess, multiprocessing


def upload_to_drive(stage=0):
    LOCAL = "data/dataset"
    REMOTE = f"drive:dataset/stage_{stage}"
    bundles = [d for d in os.listdir(LOCAL)
               if os.path.isdir(os.path.join(LOCAL, d)) and d.isdigit()]
    for bundle in bundles:
        archive = f"{bundle}.tar"
        subprocess.run(["tar", "-cf", archive, "-C", LOCAL, bundle], check=True)
        subprocess.run([
            "rclone", "copy", archive, REMOTE,
            "--checksum", "--transfers=64", "--checkers=64",
            "--fast-list", "--multi-thread-streams=4", "--progress",
        ], check=True)
        os.remove(archive)
        logging.info(f"Processed and uploaded bundle {bundle}")
