"""Prepare a pinned dataset and encoder, and remove train/evaluation text overlap."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from encoder import Encoder, ROOT


def download(url,path):
    if path.exists():
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+".part")
    with urlopen(url,timeout=180) as source,temp.open("wb") as target:
        while block:=source.read(1024*1024):
            target.write(block)
    temp.replace(path)


def normalize(text):
    return " ".join(text.casefold().split())


def load_clean_data():
    source=ROOT/"data/data_full.json"
    raw=json.loads(source.read_text())
    keys=["train","val","test","oos_val","oos_test"]
    for key in keys:
        assert len({normalize(text) for text,_ in raw[key]})==len(raw[key]),key
    evaluation={normalize(text) for key in ["val","test","oos_val","oos_test"] for text,_ in raw[key]}
    overlap=[i for i,(text,_) in enumerate(raw["train"]) if normalize(text) in evaluation]
    kept=[i for i in range(len(raw["train"])) if i not in set(overlap)]
    clean={key:raw[key] for key in keys}
    clean["train"]=[raw["train"][i] for i in kept]
    labels=sorted({label for _,label in clean["train"]})
    assert len(labels)==150
    quality={"dataset_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
             "original_train_rows":len(raw["train"]),"removed_train_source_rows":overlap,
             "clean_train_rows":len(kept),"classes":len(labels),
             "split_rows":{k:len(v) for k,v in clean.items()},
             "normalization":"Unicode casefold and whitespace collapse for overlap checking only",
             "policy":"Remove exact normalized training texts appearing in any evaluation split; keep official evaluation rows.",
             "label_order":labels}
    return clean,quality


def main():
    dataset=json.loads((ROOT/"dataset_revision.json").read_text())
    download(dataset["url"],ROOT/"data/data_full.json")
    model=json.loads((ROOT/"model_revision.json").read_text())
    base=f"https://huggingface.co/{model['model_id']}/resolve/{model['revision']}"
    for remote,local in [("onnx/model.onnx","model.onnx"),("tokenizer.json","tokenizer.json")]:
        download(base+"/"+remote,ROOT/"models/minilm"/local)
    data,quality=load_clean_data()
    artifact=ROOT/"artifacts"
    artifact.mkdir(exist_ok=True)
    (artifact/"data_quality.json").write_text(json.dumps(quality,indent=2))
    signature=hashlib.sha256(json.dumps({"data":data,"model":model},sort_keys=True).encode()).hexdigest()
    cache=artifact/"embeddings.npz"
    if cache.exists():
        with np.load(cache,allow_pickle=False) as saved:
            if saved["signature"].item()==signature:
                print("Validated embedding cache",flush=True)
                return
        raise ValueError("Embedding cache signature differs; remove the stale cache first")
    encoder=Encoder()
    arrays={}
    for split,rows in data.items():
        print("ENCODING",split,len(rows),flush=True)
        arrays[split]=encoder.encode([text for text,_ in rows],progress=True)
    np.savez_compressed(cache,signature=np.array(signature),**arrays)
    print("Prepared",quality["clean_train_rows"],"training rows; removed overlap",len(quality["removed_train_source_rows"]),flush=True)


if __name__=="__main__":
    main()
