import tensorflow as tf
import numpy as np
import os
import urllib.request

# =========================
# Load TinyStories
# =========================
TINYSTORIES_BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TINYSTORIES_TRAIN_URL = TINYSTORIES_BASE_URL + "TinyStoriesV2-GPT4-train.txt"
TINYSTORIES_VALID_URL = TINYSTORIES_BASE_URL + "TinyStoriesV2-GPT4-valid.txt"
TINYSTORIES_TRAIN_PATH = os.environ.get(
    "MORA_TINYSTORIES_TRAIN_PATH",
    os.environ.get(
        "MORA_TINYSTORIES_PATH",
        os.path.join("data", "TinyStoriesV2-GPT4-train.txt"),
    ),
)
TINYSTORIES_VALID_PATH = os.environ.get(
    "MORA_TINYSTORIES_VALID_PATH",
    os.path.join("data", "TinyStoriesV2-GPT4-valid.txt"),
)
TINYSTORIES_MAX_BYTES = int(os.environ.get("MORA_TINYSTORIES_MAX_BYTES", 25_000_000))
TINYSTORIES_VALID_MAX_BYTES = os.environ.get("MORA_TINYSTORIES_VALID_MAX_BYTES")
TINYSTORIES_VALID_MAX_BYTES = (
    int(TINYSTORIES_VALID_MAX_BYTES)
    if TINYSTORIES_VALID_MAX_BYTES
    else None
)


def ensure_tinystories(url, path, max_bytes=None):
    if os.path.exists(path):
        size = os.path.getsize(path)
        if max_bytes is None or size >= max_bytes:
            print(f"Using cached TinyStories: {path} ({size:,} bytes)")
            return path

    path_dir = os.path.dirname(path)
    if path_dir:
        os.makedirs(path_dir, exist_ok=True)

    headers = {"User-Agent": "MoraV5-TinyStories/0.1"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    request = urllib.request.Request(url, headers=headers)

    label = "all bytes" if max_bytes is None else f"{max_bytes:,} bytes"
    print(f"Downloading TinyStories to {path} ({label})")
    bytes_written = 0
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
        while True:
            read_size = 1024 * 1024
            if max_bytes is not None:
                remaining = max_bytes - bytes_written
                if remaining <= 0:
                    break
                read_size = min(read_size, remaining)

            chunk = response.read(read_size)
            if not chunk:
                break
            f.write(chunk)
            bytes_written += len(chunk)

    print(f"Loaded TinyStories: {path} ({bytes_written:,} bytes)")
    return path


def load_text_bytes(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8), len(text)


train_path = ensure_tinystories(
    TINYSTORIES_TRAIN_URL,
    TINYSTORIES_TRAIN_PATH,
    max_bytes=TINYSTORIES_MAX_BYTES,
)
valid_path = ensure_tinystories(
    TINYSTORIES_VALID_URL,
    TINYSTORIES_VALID_PATH,
    max_bytes=TINYSTORIES_VALID_MAX_BYTES,
)

train_data, train_chars = load_text_bytes(train_path)
val_data, val_chars = load_text_bytes(valid_path)
print(
    f"Loaded TinyStories train: {train_chars:,} chars; "
    f"validation: {val_chars:,} chars"
)

# Byte-level encoding
VOCAB_SIZE = 256

SAMPLE_SEPARATOR_BYTES = set(b" \t\n\r.,;:!?\"'()[]{}<>-/\\|_*&^%$#@`~+=")

SEQ_LEN = 256
BATCH_SIZE = 16
EPOCHS = 1
SAMPLE_PROMPT = os.environ.get("MORA_SAMPLE_PROMPT", "ROMEO:\n")
SAMPLE_LEN = int(os.environ.get("MORA_SAMPLE_LEN", 400))
SAMPLE_TEMPERATURE = float(os.environ.get("MORA_SAMPLE_TEMPERATURE", 0.7))
SAMPLE_TOP_K = int(os.environ.get("MORA_SAMPLE_TOP_K", 40))
GEN_CONTEXT_LEN = min(SEQ_LEN, int(os.environ.get("MORA_GEN_CONTEXT_LEN", 256)))
STREAM_SAMPLE = os.environ.get("MORA_STREAM_SAMPLE", "1") != "0"
WEIGHTS_PATH = os.environ.get("MORA_WEIGHTS_PATH", os.path.join("checkpoints", "mora.weights.h5"))

def split_input_target(chunk):
    chunk = tf.cast(chunk, tf.int32)
    return chunk[:-1], chunk[1:]

def make_dataset(data, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(data.astype(np.int32))
    dataset = dataset.batch(SEQ_LEN + 1, drop_remainder=True)
    dataset = dataset.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(10_000)

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =========================
# Delta + Importance Layer
# =========================
class DeltaImportance(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.k = self.add_weight(shape=(), initializer="ones")  # decay rate

    def call(self, h):
        # h: (B, T, D)

        # CUMSUM (memory)
        h_cum = tf.cumsum(h, axis=1)
        h_prev = tf.pad(h[:, :-1, :], [[0, 0], [1, 0], [0, 0]])

        # DELTA (change)
        delta = h_cum - h_prev

        # Importance = |delta| * exp(-k * distance)
        T = tf.shape(h)[1]
        t = tf.range(T)[None, :, None]
        j = tf.range(T)[None, None, :]
        dist = tf.cast(t - j, tf.float32)
        mask = tf.cast(j <= t, tf.float32)

        decay = tf.exp(-tf.nn.softplus(self.k) * tf.abs(dist))
        delta_mag = tf.norm(delta, axis=-1, keepdims=True)

        importance = delta_mag * decay * mask  # (B, T, T)

        # Weighted sum of past states
        context = tf.einsum("bij,bjd->bid", importance, h)

        return tf.abs(context)

class CumprodBoundaryDimension(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kw):
        super().__init__(**kw)
    def call(self, e):
        e   = tf.cumsum(tf.tanh(tf.math.cumprod(e, axis=-1)), axis=1)
        return e

# =========================
# Model
# =========================
class DeltaLM(tf.keras.Model):
    def __init__(self, dim=128):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(VOCAB_SIZE, dim)
        self.boundary_dim = CumprodBoundaryDimension(dim)
        self.blocks = [tf.keras.Sequential([
            DeltaImportance(dim),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(dim * 4, activation="gelu"),
            tf.keras.layers.Dense(dim),
            
        ]) for _ in range(3)]

    def call(self, x):
        h = self.embed(x)               # (B, T, D)
        h = h + h * self.boundary_dim(h)    # (B, T, D)
        for block in self.blocks:
            h += block(h)
        return tf.matmul(h, self.embed.embeddings, transpose_b=True)


# =========================
# Train
# =========================
model = DeltaLM(dim=64)
optimizer = tf.keras.optimizers.Adam(3e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def bits_per_byte(y_true, y_pred):
    loss = tf.keras.backend.sparse_categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=True,
    )
    return tf.reduce_mean(loss) / tf.math.log(tf.constant(2.0, dtype=loss.dtype))

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy', bits_per_byte]
)

history = model.fit(
    make_dataset(train_data, shuffle=True),
    validation_data=make_dataset(val_data),
    epochs=EPOCHS,
)

if "val_bits_per_byte" in history.history:
    print(f"Final validation BPB: {history.history['val_bits_per_byte'][-1]:.4f}")

weights_dir = os.path.dirname(WEIGHTS_PATH)
if weights_dir:
    os.makedirs(weights_dir, exist_ok=True)
model.save_weights(WEIGHTS_PATH)
print(f"Saved model weights to {WEIGHTS_PATH}")

# =========================
# Sample
# =========================
def _sample_next_byte(model, context, temperature=0.7, top_k=40):
    logits = model(context)[:, -1, :]
    previous_byte = int(context[0, -1])

    if previous_byte in SAMPLE_SEPARATOR_BYTES:
        scaled_logits = logits / max(temperature, 1e-6)
        values, indices = tf.math.top_k(scaled_logits, k=min(top_k, VOCAB_SIZE))
        choice = tf.random.categorical(values, num_samples=1)
        return int(indices[0, choice[0, 0]].numpy())

    return int(tf.argmax(logits[0]).numpy())


def sample_bytes(
    model,
    start=SAMPLE_PROMPT,
    length=SAMPLE_LEN,
    temperature=SAMPLE_TEMPERATURE,
    top_k=SAMPLE_TOP_K,
    context_len=GEN_CONTEXT_LEN,
):
    tokens = list(start.encode("utf-8"))
    yield from tokens

    for _ in range(length):
        context = np.array(tokens[-context_len:], dtype=np.int32)[None, :]
        next_token = _sample_next_byte(model, context, temperature=temperature, top_k=top_k)
        tokens.append(next_token)
        yield next_token


def sample(model, **kwargs):
    generated = bytearray(sample_bytes(model, **kwargs))
    return generated.decode("utf-8", errors="ignore")


def stream_sample(model):
    for token in sample_bytes(model):
        print(bytes([token]).decode("utf-8", errors="ignore"), end="", flush=True)
    print()


print(f"\n=== SAMPLE (context={GEN_CONTEXT_LEN}, stream={STREAM_SAMPLE}) ===\n")
if STREAM_SAMPLE:
    stream_sample(model)
else:
    print(sample(model))

model.summary()