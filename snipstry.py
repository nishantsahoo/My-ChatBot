import io
import json
from snips_nlu import SnipsNLUEngine

with io.open("train.json") as f:
    sample_dataset = json.load(f)

nlu_engine = SnipsNLUEngine()

nlu_engine.fit(sample_dataset)

parsing = nlu_engine.parse(u"Tell me in detail about Nishant?")
print(json.dumps(parsing, indent=2))