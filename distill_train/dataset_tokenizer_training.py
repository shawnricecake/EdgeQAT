from pathlib import Path
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC

# We train the tokenizer on the train data only
data_dir = Path("/path/babylm_10M_clean")

paths = [str(f) for f in data_dir.glob("*") if f.is_file() and not f.name.endswith(".DS_Store") and f.suffix in [".train"]]

# paths
print(len(paths))
assert len(paths) > 0, 'No data files found'

tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
tokenizer.normalizer = NFKC()

trainer = trainers.BpeTrainer(vocab_size=16000, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
tokenizer.train(paths, trainer)

tokenizer_path =  "/path/gpt-clean-16000.json"
tokenizer.save(str(tokenizer_path), pretty=True)

tokenizer = Tokenizer.from_file(str(tokenizer_path))


# TESTING:
# text = 'Shiro Okada (岡田志郎, "Okada Shirō", June 9, 1949; Hirakata, Osaka {age 71} - ) is a Japanese guitarist who participate in the Group Sound band, the Ox. His nickname was Shiro (シロー) and his real name is Shiro Okamoto (岡田史郎).'
text = "The quick brown fox jumps over the lazy dog."

encoded = tokenizer.encode(text)
print(f"Encoded String: {encoded.tokens}")

print(f"Encoded IDs: {encoded.ids}")

decoded = tokenizer.decode(encoded.ids)
print(f"Decoded String: {decoded}")

