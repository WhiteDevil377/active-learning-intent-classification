"""Frozen MiniLM embeddings with attention-masked mean pooling."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent


class Encoder:
    def __init__(self):
        model = ROOT / "models/minilm"
        self.tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model / "model.onnx"), sess_options=opts,
                                           providers=["CPUExecutionProvider"])
        self.inputs = {x.name for x in self.session.get_inputs()}

    def encode(self, texts, batch_size=64, progress=False):
        if not texts:
            raise ValueError("At least one text is required")
        batches = []
        for start in range(0, len(texts), batch_size):
            items = self.tokenizer.encode_batch(texts[start:start+batch_size])
            data = {"input_ids": np.array([x.ids for x in items],dtype=np.int64),
                    "attention_mask": np.array([x.attention_mask for x in items],dtype=np.int64),
                    "token_type_ids": np.array([x.type_ids for x in items],dtype=np.int64)}
            hidden = self.session.run(None,{k:v for k,v in data.items() if k in self.inputs})[0]
            mask = data["attention_mask"][...,None].astype(np.float32)
            pooled = (hidden*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1e-9)
            pooled /= np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-9)
            batches.append(pooled.astype(np.float32))
            if progress and (start==0 or (start//batch_size+1)%50==0):
                print(f"Embedded {min(start+batch_size,len(texts))}/{len(texts)}",flush=True)
        return np.vstack(batches)
