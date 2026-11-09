"""Predict intent and apply the validation-selected confidence gate."""
import argparse
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from encoder import Encoder, ROOT


class Predictor:
    def __init__(self):
        self.bundle=joblib.load(ROOT/"artifacts/classifier.joblib")
        if self.bundle["model_revision"]!=json.loads((ROOT/"model_revision.json").read_text()):
            raise ValueError("Classifier and encoder revisions do not match")
        self.encoder=Encoder()

    def predict(self,texts):
        if not isinstance(texts,list) or not 1<=len(texts)<=32:
            raise ValueError("Provide between 1 and 32 texts")
        if any(not isinstance(text,str) or not text.strip() or len(text)>1000 for text in texts):
            raise ValueError("Each text must be a nonempty string of at most 1000 characters")
        with threadpool_limits(limits=2):
            probabilities=self.bundle["classifier"].predict_proba(self.encoder.encode(texts))
        result=[]
        for text,p in zip(texts,probabilities):
            top=np.argsort(-p,kind="stable")[:3]
            rejected=bool(p[top[0]]<self.bundle["threshold"])
            result.append({"text":text,"intent":"out_of_scope" if rejected else self.bundle["labels"][int(top[0])],
                           "rejected":rejected,"confidence":float(p[top[0]]),"threshold":self.bundle["threshold"],
                           "top_known_intents":[{"intent":self.bundle["labels"][int(i)],"score":float(p[i])} for i in top]})
        return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text",nargs="+")
    args=parser.parse_args()
    print(json.dumps(Predictor().predict(args.text),indent=2))
