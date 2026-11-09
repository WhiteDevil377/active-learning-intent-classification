"""Label-selection strategies; acquisition functions never receive hidden labels."""
from collections import deque
import numpy as np


def select_batch(probabilities,available,count,strategy,rng):
    available=np.asarray(available,dtype=int)
    if not 0<count<=len(available):
        raise ValueError("Invalid acquisition batch size")
    if strategy=="random":
        return rng.choice(available,size=count,replace=False)
    p=np.asarray(probabilities)
    if len(p)!=len(available) or p.ndim!=2:
        raise ValueError("Probabilities must match the available pool")
    if strategy=="entropy":
        uncertainty=-(p*np.log(np.maximum(p,1e-12))).sum(axis=1)
        chosen=np.argsort(-uncertainty,kind="stable")[:count]
    elif strategy=="balanced_margin":
        top2=np.partition(p,-2,axis=1)[:,-2:]
        gap=top2.max(axis=1)-top2.min(axis=1)
        order=np.argsort(gap,kind="stable")
        predicted=p.argmax(axis=1)
        groups={int(label):deque(order[predicted[order]==label].tolist()) for label in np.unique(predicted)}
        # Round-robin across predicted classes; within each class query smallest margins first.
        result=[]
        while len(result)<count:
            for label in sorted(groups):
                if groups[label]:
                    result.append(groups[label].popleft())
                    if len(result)==count:
                        break
        chosen=np.array(result,dtype=int)
    else:
        raise ValueError("Unknown strategy")
    return available[chosen]
