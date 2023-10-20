import json
import os  # Import the os module
import shutil
import subprocess
from io import BytesIO
from typing import List

import boto3
import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

BUCKET_NAME = "animeme-v0"  # Define as class-level attribute
REGION = "us-east-1"  # Define as class-level attribute

class LoadFromS3:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_key": ("STRING", {"default": "path/to/your/image.jpg"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image"
    CATEGORY = "Ikhor"

    def load_image(self, file_key):
        # Retrieve AWS credentials from environment variables
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables.")

        # Initialize S3 client
        s3 = boto3.client('s3', region_name=REGION,
                          aws_access_key_id=aws_access_key,
                          aws_secret_access_key=aws_secret_key)

        # Fetch the image from S3
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
        img_data = BytesIO(obj['Body'].read())

        i = Image.open(img_data)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask.unsqueeze(0))

class LoadBatchFromS3:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "test-james"}),
                "max_images": ("INT", {"default": 32})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    FUNCTION = "load_all_images"
    CATEGORY = "Ikhor"

    def load_all_images(self, folder_path, max_images):
        # Extract bucket name and folder name (prefix) from the folder_url

        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials not found in environment variables.")

        s3 = boto3.client('s3', region_name=REGION,
                          aws_access_key_id=aws_access_key,
                          aws_secret_access_key=aws_secret_key)

        # List all objects in the folder (prefix)
        objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_path)
        # Filter only .png files and limit by max_images
        file_keys = [content['Key'] for content in objects.get('Contents', []) if content['Key'].endswith('.png')][:max_images]

        images = []
        masks = []

        for key in file_keys:
            obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            img_data = BytesIO(obj['Body'].read())

            i = Image.open(img_data)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

            images.append(image)
            masks.append(mask.unsqueeze(0))

        return (torch.cat(images, dim=0), torch.stack(masks, dim=0), len(images))


class SaveGifToS3:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_path = shutil.which("ffmpeg")
        #Hide ffmpeg formats if ffmpeg isn't available
        if ffmpeg_path is not None:
            ffmpeg_formats = ["video/"+x[:-5] for x in folder_paths.get_filename_list("video_formats")]
        else:
            ffmpeg_formats = []
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "max": 24, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "s3_folder": ("STRING", {"default": "test-default-folder"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_image": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("GIF",)
    OUTPUT_NODE = True
    CATEGORY = "Ikhor"
    FUNCTION = "generate_and_upload_video"

    def generate_and_upload_video(
        self,
        images,
        frame_rate: int,
        loop_count: int,
        s3_folder="test-folder",
        format="image/gif",
        pingpong=False,
        save_image=True,
        prompt=None,
        extra_pnginfo=None,
    ):

        print("saving video to s3")
        # convert images to numpy
        frames: List[Image.Image] = []
        frame_counter = 0
        for image in images:
            print(f"saving frame {frame_counter}")
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            frames.append(img)
            frame_counter += 1

        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_image
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path("final", output_dir)
        print(f"output directory: {output_dir}")
        metadata = PngInfo()
#         if prompt is not None:
#             metadata.add_text("prompt", json.dumps(prompt))
#         if extra_pnginfo is not None:
#             for x in extra_pnginfo:
#                 metadata.add_text(x, json.dumps(extra_pnginfo[x]))
#
#         # save first frame as png to keep metadata
#         file = f"{filename}_{counter:05}_.png"
#         file_path = os.path.join(full_output_folder, file)
#         frames[0].save(
#             file_path,
#             pnginfo=metadata,
#             compress_level=4,
#         )
        if pingpong:
            frames = frames + frames[-2:0:-1]

        format_type, format_ext = format.split("/")
        file = f"{filename}_{counter:05}_.{format_ext}"
        file_path = os.path.join(full_output_folder, file)
        if format_type == "image":
            # Use pillow directly to save an animated image
            frames[0].save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames[1:],
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
            )
        else:
            # Use ffmpeg to save a video
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                #Should never be reachable
                raise ProcessLookupError("Could not find ffmpeg")

            video_format_path = folder_paths.get_full_path("video_formats", format_ext + ".json")
            with open(video_format_path, 'r') as stream:
                video_format = json.load(stream)
            file = f"{filename}_{counter:05}_.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            dimensions = f"{frames[0].width}x{frames[0].height}"
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + video_format['main_pass'] + [file_path]

            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])
            with subprocess.Popen(args, stdin=subprocess.PIPE, env=env) as proc:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())

            s3_file_path = f"{s3_folder}/final.mp4"  # Replace 'desired_path_on_s3' with your desired path

            s3 = boto3.client('s3',
                              region_name=REGION,  # Adjust if using a different region
                              aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                              aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

            with open(file_path, 'rb') as file_data:
                s3.put_object(Bucket=BUCKET_NAME, Key=s3_file_path, Body=file_data, ACL='public-read')
            print("Completed uploading s3 object")
        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_image else "temp",
                "format": format,
            }
        ]
        return {"ui": {"gifs": previews}}


# Add this new node to the dictionary of all nodes
NODE_CLASS_MAPPINGS = {
    "LoadFromS3": LoadFromS3,
    "LoadBatchFromS3": LoadBatchFromS3,
    "SaveGifToS3": SaveGifToS3
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadBatchFromS3": "Load Image Batch from S3",
    "LoadFromS3": "Load Image from S3",
    "SaveGifToS3": "Save GIF to S3"
}
