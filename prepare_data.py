import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parent
REVISION = "f7e23e2b381b0120e88222124f0b06baac432630"
ARCHIVE_URL = f"https://codeload.github.com/NWPU-903PR/HyperSynergy/zip/{REVISION}"
ARCHIVE_ROOT = f"HyperSynergy-{REVISION}"


def upstream_path(relative):
    return relative.replace("drugcomb_few_shot/", "data_few_zero_rich/", 1)


def normalized(data, relative):
    # Upstream checkouts may convert the line endings in TSV files.
    return data.replace(b"\r\n", b"\n") if relative.endswith(".txt") else data


def prepare(read_source, destination):
    manifest = (ROOT / "data" / "sha256.txt").read_text(encoding="utf-8").splitlines()
    verified = {}
    for line in manifest:
        digest, relative = line.split("  ", 1)
        content = normalized(read_source(upstream_path(relative)), relative)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Source checksum mismatch for {relative}")
        target = destination / relative
        if target.exists():
            existing = normalized(target.read_bytes(), relative)
            if hashlib.sha256(existing).hexdigest() != digest:
                raise ValueError(f"Existing file differs from the benchmark: {target}")
        else:
            verified[target] = content

    for target, content in verified.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(target)
    print(f"Verified {len(manifest)} data files in {destination}")


def main():
    parser = argparse.ArgumentParser(description="Prepare the DiffPro benchmark data")
    parser.add_argument("--source", type=Path, help="data directory of a local HyperSynergy checkout")
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    try:
        if args.source is not None:
            prepare(lambda relative: (args.source / relative).read_bytes(), args.destination)
        else:
            with tempfile.TemporaryFile() as downloaded:
                print(f"Downloading HyperSynergy data at {REVISION}")
                with urlopen(ARCHIVE_URL, timeout=60) as response:
                    shutil.copyfileobj(response, downloaded)
                downloaded.seek(0)
                with ZipFile(downloaded) as archive:
                    prepare(
                        lambda relative: archive.read(f"{ARCHIVE_ROOT}/data/{relative}"),
                        args.destination,
                    )
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")


if __name__ == "__main__":
    main()
