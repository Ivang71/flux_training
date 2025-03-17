import os, asyncio, base64, sys, pickle, json, logging
from huggingface_hub import login
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from os.path import join
import nest_asyncio
nest_asyncio.apply()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

k = base64.b64decode('aGZfaHZqck9VTXFvTXF3dW9HR3JoTlZKSWlsZUtFTlNQbXRjTw==').decode()
login(token=k, add_to_git_credential=False)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = k
os.environ['HF_TOKEN'] = k
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"


pipe = pipeline('OpenGVLab/InternVL2_5-26B-MPO-AWQ', backend_config=TurbomindEngineConfig(device_type='cuda', session_len=14336))
prompt = """
Describe the image in detail, STRICTLY keeping it under 100-150 words, in a single coherent paragraph.
DO NOT BEGIN WITH 'This image shows', 'In this image', 'The image depicts', etc. Clearly specify the gender (man, woman, boy, girl, or male/female).
Focus on the character's appearance, gestures, poses, clothing, and any accessories they might be wearing.
If the character is interacting with any objects, describe the nature of the interaction clearly, including interactions with the background.
Describe the character's expressions and emotions.
Mention any text visible on the character or objects (e.g., logos, patterns, or labels).
Specify the lighting’s direction, intensity, and its effect on the character (e.g., shadows or highlights on the body or clothing).
Indicate the style of the image (e.g., cartoon, photograph, 3D render) and avoid adding subjective interpretations or speculation. Keep the description strictly factual and focus solely on the observable details within the image.
"""


async def process_char_dir(char_dir):
    logging.info(f"Processing character directory: {char_dir}")
    pkl_path = join(char_dir, "char_data.pkl")
    with open(pkl_path, "rb") as f:
        char_data = pickle.load(f)
    
    if all("label" in item for item in char_data):
        logging.info(f"All labels present in {char_dir}; skipping.")
        return 0

    img_paths = sorted([join(char_dir, f) for f in os.listdir(char_dir) if f.lower().endswith('.jpg')])
    logging.info(f"Found {len(img_paths)} images in {char_dir}")
    tasks = [asyncio.to_thread(load_image, img_path) for img_path in img_paths]
    imgs = await asyncio.gather(*tasks)
    prompts = [(prompt, img) for img in imgs]

    labels = []
    for i in range(0, len(prompts), 10):
        batch = prompts[i:i+10]
        logging.info(f"Processing batch {i//10} in {char_dir}")
        # Assuming pipe returns objects with a 'text' attribute.
        labels_batch = [res.text for res in pipe(batch)]
        logging.info(f"Batch {i//10} produced {len(labels_batch)} labels")
        labels.extend(labels_batch)
    
    for i, lab in enumerate(labels):
        if i < len(char_data):
            char_data[i]["label"] = lab
        else:
            logging.warning(f"No corresponding char_data element for index {i} in {char_dir}")
    
    with open(pkl_path, "wb") as f:
        pickle.dump(char_data, f)
    logging.info(f"Finished processing {char_dir}")
    return 0



async def process_bundle(bundle):
    logging.info(f"Processing bundle: {bundle}")
    info_path = join(bundle, "info.json")
    if os.path.exists(info_path):
        # Load and convert any existing labeled indices to integers
        info = json.load(open(info_path))
        info["labeled"] = [int(x) for x in info.get("labeled", [])]
    else:
        info = {"all_labeled": False, "labeled": []}    
    if info.get("all_labeled"):
        logging.info(f"Bundle {bundle} is fully labeled; skipping.")
        return    
    # Process character directories sorted numerically.
    for d in sorted([d for d in os.listdir(bundle) if os.path.isdir(join(bundle, d)) and d.isdigit()], key=lambda d: int(d)):
        num_d = int(d)
        if num_d not in info["labeled"]:
            char_path = join(bundle, d)
            logging.info(f"Processing character {d} in bundle {bundle}")
            await process_char_dir(char_path)
            info["labeled"].append(num_d)
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)   
    if 999 in info["labeled"]:
        info["all_labeled"] = True    
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logging.info(f"Final bundle info for {bundle}: {info}")



async def poll_bundles(base):
    while True:
        for d in os.listdir(base):
            bundle = join(base, d)
            if os.path.isdir(bundle):
                await process_bundle(bundle)
        logging.info("Nothing to process; waiting 13 seconds for updates.")
        await asyncio.sleep(13)


async def main():
    base_dir = "./data/dataset"
    logging.info("Starting bundle polling.")
    await poll_bundles(base_dir)


if __name__ == "__main__":
    asyncio.run(main())
