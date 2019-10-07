import io
import sys
import json
from snips_nlu import SnipsNLUEngine
import bs4

with io.open("train.json") as f:
    sample_dataset = json.load(f)

nlu_engine = SnipsNLUEngine()

nlu_engine.fit(sample_dataset)

parsing = nlu_engine.parse(str(" ".join(sys.argv[1:])))

print(json.dumps(parsing, indent = 4))