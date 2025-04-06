import os
import json
from torch.utils.data import Dataset
from PIL import Image


class COCODataset(Dataset):
    def __init__(
        self, image_dir, annotation_path, transform=None, tokenizer=None
    ) -> None:
        super(COCODataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

        with open(annotation_path, "r") as f:
            annotations = json.load(f)

        self.id_to_filename = {
            img["id"]: img["file_name"] for img in annotations["images"]
        }
        self.samples = [
            {
                "image_path": os.path.join(
                    image_dir, self.id_to_filename[ann["image_id"]]
                ),
                "caption": ann["caption"],
            }
            for ann in annotations["annotations"]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = item["caption"]

        if self.tokenizer:
            caption = self.tokenizer(
                caption, padding="max_length", truncation=True, return_tensors="pt"
            )

        return {"image": image, "caption": caption}
