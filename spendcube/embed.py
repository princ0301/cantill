"""Optional local sentence embeddings (no API). Gives the zero-label classifier real semantics.

Backends:
  sentence-transformers   pip/uv: sentence-transformers   (pulls in torch; larger install)
  fastembed               pip/uv: fastembed               (ONNX; much lighter)
  hash                    built in, NOT semantic (character n-gram hashing); only for tests/offline checks
The model is downloaded on first use, so run it once with internet before the blind test.
"""
import hashlib
import numpy as np


class EmbeddingsUnavailable(Exception):
    pass


def _normalise(a):
    a = np.asarray(a, dtype="float32")
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


class HashEmbedder:
    def __init__(self, dim=256, **_):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            t = f"  {str(t).lower()}  "
            for j in range(len(t) - 2):
                h = int(hashlib.md5(t[j:j + 3].encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        return _normalise(out)


class Embedder:
    def __init__(self, backend="sentence-transformers", model="paraphrase-multilingual-MiniLM-L12-v2", batch=64):
        self.backend, self.model_name, self.batch = backend, model, batch
        self._m = None

    def _load(self):
        if self._m is not None:
            return
        try:
            if self.backend == "sentence-transformers":
                from sentence_transformers import SentenceTransformer
                self._m = SentenceTransformer(self.model_name)
            elif self.backend == "fastembed":
                from fastembed import TextEmbedding
                self._m = TextEmbedding(model_name=self.model_name)
            else:
                raise EmbeddingsUnavailable(f"unknown embeddings backend '{self.backend}'")
        except ImportError as e:
            pkg = "sentence-transformers" if self.backend == "sentence-transformers" else "fastembed"
            raise EmbeddingsUnavailable(f"embeddings need the '{pkg}' package: uv add {pkg}  ({e})")
        except EmbeddingsUnavailable:
            raise
        except Exception as e:                          # download / offline problems
            raise EmbeddingsUnavailable(f"could not load embedding model '{self.model_name}' "
                                        f"(needs internet on first use): {e}")

    def encode(self, texts):
        self._load()
        texts = [str(t) if str(t).strip() else " " for t in texts]
        uniq = list(dict.fromkeys(texts))
        if self.backend == "sentence-transformers":
            vec = self._m.encode(uniq, batch_size=self.batch, show_progress_bar=False, normalize_embeddings=True)
        else:
            vec = np.array(list(self._m.embed(uniq)))
        vec = _normalise(vec)
        pos = {t: i for i, t in enumerate(uniq)}
        return vec[[pos[t] for t in texts]]


def make_embedder(cfg_block):
    """Build an embedder from the config / CLI settings (a dict with backend, model)."""
    b = (cfg_block or {}).get("backend", "sentence-transformers")
    if b == "hash":
        return HashEmbedder()
    default = {"sentence-transformers": "paraphrase-multilingual-MiniLM-L12-v2",
               "fastembed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}.get(b, "")
    return Embedder(b, (cfg_block or {}).get("model", default))