"""Validate acquisition isolation, trajectories, predictor behavior, and the real API."""
import inspect
import json
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import numpy as np
from acquisition import select_batch
from encoder import Encoder,ROOT
from predict import Predictor
from prepare import load_clean_data,normalize


def main():
    assert "labels" not in inspect.signature(select_batch).parameters
    available=np.array([10,20,30,40])
    p=np.array([[.8,.2],[.51,.49],[.95,.05],[.5,.5]])
    assert select_batch(p,available,1,"entropy",np.random.default_rng(42)).tolist()==[40]
    first=select_batch(None,available,3,"random",np.random.default_rng(42))
    second=select_batch(None,available,3,"random",np.random.default_rng(42))
    assert len(set(first))==3 and set(first)<=set(available)
    np.testing.assert_array_equal(first,second)
    diverse=np.array([[.51,.49],[.50,.50],[.49,.51],[.4,.6]])
    ids=select_batch(diverse,np.arange(4),2,"balanced_margin",np.random.default_rng(42))
    assert set(diverse[ids].argmax(axis=1))=={0,1}
    data,quality=load_clean_data()
    training={normalize(text) for text,_ in data["train"]}
    evaluation={normalize(text) for key in ["val","test","oos_val","oos_test"] for text,_ in data[key]}
    assert not training&evaluation
    trajectories=json.loads((ROOT/"artifacts/acquisition_indices.json").read_text())
    for seed in [42,43,44,45,46]:
        starters=[]
        for strategy in ["random","entropy","balanced_margin"]:
            previous=set()
            for budget in [450,900,1500,3000]:
                ids=trajectories[f"{seed}_{strategy}_{budget}"]
                assert len(ids)==budget and len(set(ids))==budget
                assert previous<=set(ids) and min(ids)>=0 and max(ids)<len(data["train"])
                if budget==450:
                    starters.append(ids)
                previous=set(ids)
        assert starters[0]==starters[1]==starters[2]
    predictor=Predictor()
    assert len(predictor.bundle["labels"])==150
    np.testing.assert_array_equal(predictor.bundle["classifier"].classes_,np.arange(150))
    text="what is my account balance"
    results=predictor.predict([text,"hello, can you help me"])
    assert len(results)==2
    for item in results:
        assert 0<=item["confidence"]<=1 and len(item["top_known_intents"])==3
        assert item["rejected"]==(item["confidence"]<item["threshold"])
        assert item["intent"]=="out_of_scope" or item["intent"] in predictor.bundle["labels"]
    for invalid in [[],None,[""],[1],["x"*1001]]:
        try:
            predictor.predict(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid text input accepted")
    a=predictor.encoder.encode([text])[0]
    b=predictor.encoder.encode([text,"A longer sentence. "*100])[0]
    np.testing.assert_allclose(a,b,atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(np.linalg.norm(a),1,atol=1e-5)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    process=subprocess.Popen([sys.executable,str(ROOT/"serve.py"),"--port",str(port)],
                             cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    base=f"http://127.0.0.1:{port}"
    try:
        ready=False
        for _ in range(80):
            if process.poll() is not None:
                raise RuntimeError(process.stderr.read().decode())
            try:
                with urlopen(base+"/health",timeout=.5) as response:
                    assert json.load(response)["intents"]==150
                ready=True
                break
            except OSError:
                time.sleep(.25)
        assert ready,"API startup timed out"
        request=Request(base+"/predict",data=json.dumps({"texts":[text]}).encode(),
                        headers={"Content-Type":"application/json"},method="POST")
        with urlopen(request,timeout=10) as response:
            assert len(json.load(response)["predictions"])==1
        invalid=Request(base+"/predict",data=b'{"texts": [""]}',
                        headers={"Content-Type":"application/json"},method="POST")
        try:
            urlopen(invalid,timeout=5)
        except HTTPError as error:
            assert error.code==400
        else:
            raise AssertionError("API accepted an empty text")
    finally:
        process.terminate()
        process.communicate(timeout=10)
    print("PASS: acquisition isolation, entropy query, balanced batches, equal starters, nested budgets, decontamination, 150-class probabilities, rejection gate, masked pooling, live API and invalid input")


if __name__=="__main__":
    main()
