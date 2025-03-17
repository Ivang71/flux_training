import os, asyncio, base64, sys, pickle, time, json
import numpy as np
from PIL import Image
from huggingface_hub import login
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
import nest_asyncio
nest_asyncio.apply()


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
    pkl_path = os.path.join(char_dir, "char_data.pkl")
    with open(pkl_path, "rb") as f:
        char_data = pickle.load(f)

    if all("label" in item for item in char_data):
        return 0

    img_paths = [os.path.join(char_dir, f) for f in os.listdir(char_dir) if f.lower().endswith(('.jpg'))]
    img_paths.sort()
    tasks = [asyncio.to_thread(load_image, img_path) for img_path in img_paths]
    imgs = await asyncio.gather(*tasks)
    prompts = [(prompt, img) for img in imgs]

    labels = []
    for i in range(0, len(prompts), 10):
        labels_batch = [res.text for res in pipe(prompts[i:i+10])]
        print(labels_batch)
        labels.extend(labels_batch)
    
    for i, label in enumerate(labels):
        if i < len(char_data):
            char_data[i]["label"] = label
        else:
            print(f"Warning: No corresponding char_data element for index {i}")

    # for item in char_data:
    #     item["face_embedding"] = []
    #     item["body_embedding"] = []

    # def default_converter(o):
    #     if isinstance(o, np.ndarray):
    #         return o.tolist()
    #     raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    # print(json.dumps(char_data, default=default_converter))
    # await asyncio.sleep(2000)
    
    with open(pkl_path, "wb") as f:
        pickle.dump(char_data, f)
    return 0


async def main():
    base_dir = "./data/dataset"
    # Traverse each bundle directory
    for bundle in os.listdir(base_dir):
        bundle_dir = os.path.join(base_dir, bundle)
        if os.path.isdir(bundle_dir):
            # Traverse each character directory within the bundle
            for char in os.listdir(bundle_dir):
                char_dir = os.path.join(bundle_dir, char)
                if os.path.isdir(char_dir) and "char_data.pkl" in os.listdir(char_dir):
                    print(f"Processing {char_dir}...")
                    await process_char_dir(char_dir)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
