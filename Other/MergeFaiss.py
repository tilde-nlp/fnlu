import sys
import os
import argparse
import json

from langchain.embeddings.base import Embeddings
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import FAISS

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('vec_stores', help='file with vector stores to merge')
    parser.add_argument('out_model', help='model name to output')

    args = parser.parse_args()

    with open(args.vec_stores, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            nameprefix=line.strip()
            if i == 0:
                db0 = FAISS.load_local(nameprefix, FakeEmbeddings(size=100))
                with open(f"{nameprefix}.intentmap.json", "r", encoding="utf-8") as intentf:
                    int0dict = json.load(intentf)
            else:
                db1 = FAISS.load_local(nameprefix, FakeEmbeddings(size=100))
                db0.merge_from(db1)
                
                with open(f"{nameprefix}.intentmap.json", "r", encoding="utf-8") as intentf:
                    int1dict = json.load(intentf)
                
                maxidx=max(int0dict.values())
                maxidx = maxidx + 1
                for j, item in enumerate(int1dict):
                    if item not in int0dict:
                        int0dict[item]=maxidx+j
                    
    db0.save_local(args.out_model)
    
    with open(f"{args.out_model}.intentmap.json", "w", encoding="utf-8") as intentf:
        json.dump(int0dict, intentf, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    main()



