"""Download the checksum-pinned MVTec AD bottle mirror for noncommercial research."""

import hashlib
import tarfile
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/XavierZXY/MvT-tar/resolve/main/bottle.tar.xz"
SHA256 = "726512129cb3b1f47d4f6cd7c7fc9170db7fc209132249c788fec06029cb5dc3"


def extract_archive(archive, destination):
    def writable(member, path):
        # Reject paths and links outside the destination
        member = tarfile.data_filter(member, path)
        if member is not None:
            member.mode = 0o755 if member.isdir() else 0o644
        return member

    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter=writable)


def main():
    directory = Path("data")
    directory.mkdir(exist_ok=True)
    archive = directory / "bottle.tar.xz"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        urllib.request.urlretrieve(URL, temporary)
        temporary.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Archive checksum mismatch; no files extracted")
    extract_archive(archive, directory)
    print("Verified and extracted data/bottle")


if __name__ == "__main__":
    main()
