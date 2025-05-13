# From https://github.com/AI4LIFE-GROUP/SpLiCE

import os
import urllib

SUPPORTED_VOCAB = [
    "laion",
    "laion_bigrams",
    "mscoco"
]

GITHUB_HOST_LINK = "https://raw.githubusercontent.com/AI4LIFE-GROUP/SpLiCE/main/data/"


def _download(url: str, root: str, subfolder: str):
    root_subfolder = os.path.join(root, subfolder)
    os.makedirs(root_subfolder, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root_subfolder, filename)

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)
    return download_target


def get_vocabulary(name: str, vocabulary_size: int, download_root = None):
    if name in SUPPORTED_VOCAB:
        vocab_path = _download(os.path.join(GITHUB_HOST_LINK, "vocab", name + ".txt"), download_root or os.path.expanduser("~/.cache/splice/"), "vocab")

        vocab = []
        with open(vocab_path, "r") as f:
            lines = f.readlines()
            if vocabulary_size > 0:
                lines = lines[-vocabulary_size:]
            for line in lines:
                vocab.append(line.strip())
        return vocab
    else:
        raise RuntimeError(f"Vocabulary {name} not supported.")