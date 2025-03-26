import asyncio
import aiofiles
import shutil
import sys
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

async def copy_file_async(src: Path, dst: Path) -> None:
    """
    Asynchronously copy a single file from source to destination.

    Args:
        src (Path): Source file path
        dst (Path): Destination file path

    Raises:
        OSError: If file operations fail
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.stat().st_size < 10 * 1024 * 1024:  # Less than 10MB
            async with aiofiles.open(src, 'rb') as fsrc:
                async with aiofiles.open(dst, 'wb') as fdst:
                    await fdst.write(await fsrc.read())
        else:
            await asyncio.to_thread(shutil.copy2, src, dst)

    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {str(e)}")
        raise

async def copy_artifacts_async(source_dir: str, dest_dir: str, max_concurrent: Optional[int] = 5) -> None:
    """
    Asynchronously copy all contents from source directory to destination directory.

    Args:
        source_dir (str): Source directory path
        dest_dir (str): Destination directory path
        max_concurrent (Optional[int]): Maximum number of concurrent copy operations. Defaults to 5.

    Raises:
        FileNotFoundError: If source directory doesn't exist
        OSError: If file operations fail
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")

    dest_path.mkdir(parents=True, exist_ok=True)

    files_to_copy = []
    for src_file in source_path.rglob('*'):
        if src_file.is_file():
            rel_path = src_file.relative_to(source_path)
            dst_file = dest_path / rel_path
            files_to_copy.append((src_file, dst_file))

    semaphore = asyncio.Semaphore(max_concurrent)

    async def copy_with_semaphore(src: Path, dst: Path) -> None:
        async with semaphore:
            await copy_file_async(src, dst)

    tasks = [
        copy_with_semaphore(src, dst)
        for src, dst in files_to_copy
    ]

    await asyncio.gather(*tasks)