from pathlib import Path
from dataset_mrclean import *

DATA_ROOT = Path("/path/dataset/")
SEQ_LENGTH = 128  # this is a legacy parameter, it does not affect cleaning
DATA_SPLITS = ['babylm_10M', 'babylm_dev']

CLEANUP_FUNCTIONS = {
    'aochildes': cleanup_aochildes,
    'bnc_spoken': cleanup_bnc_spoken,
    'cbt': cleanup_cbt,
    'children_stories': cleanup_children_stories,
    'gutenberg': cleanup_gutenberg,
    'open_subtitles': cleanup_open_subtitles,
    'qed': cleanup_qed,
    'simple_wikipedia': cleanup_simple_wikipedia,
    'switchboard': cleanup_switchboard,
    'wikipedia': cleanup_wikipedia,
}

for split in DATA_SPLITS:
    INPUT_DIR = DATA_ROOT / 'babylm_data' / split
    OUTPUT_DIR = DATA_ROOT / 'babylm_data' / f'{split}_clean'

    OUTPUT_DIR.mkdir(exist_ok=True)

    train_files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix in ['.train', '.dev']]

    for file in train_files:
        text = file.read_text()
        cleaned_text = CLEANUP_FUNCTIONS[file.stem](text, SEQ_LENGTH)
        (OUTPUT_DIR / file.name).write_text(cleaned_text)
        print(f"🧹 Cleaned '{file.name}' (size {len(text)} -> {len(cleaned_text)}) in {split}")