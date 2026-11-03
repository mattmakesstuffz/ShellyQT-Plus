import os
import sentencepiece as spm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(BASE, "data", "train.txt")
OUTPUT = os.path.join(BASE, "tokenizer", "shelly")

VOCAB_SIZE = 2000

if not os.path.exists(DATA):
    raise FileNotFoundError(DATA)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("ShellyQT+ Tokenizer")
print("===================")
print("Dataset:", DATA)
print("Vocabulary size:", VOCAB_SIZE)

spm.SentencePieceTrainer.train(
    input=DATA,
    model_prefix=OUTPUT,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    shuffle_input_sentence=True,
    max_sentence_length=4096,
    num_threads=2
)

print("\nTokenizer created!")
print("Model:", OUTPUT + ".model")
print("Vocabulary:", OUTPUT + ".vocab")
