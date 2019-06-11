import io
import json

with io.open("sample_dataset.json") as f:
    sample_dataset = json.load(f)

from snips_nlu import SnipsNLUEngine

nlu_engine = SnipsNLUEngine()

nlu_engine.fit(sample_dataset)

import json

parsing = nlu_engine.parse(u"Lights on!")
print(json.dumps(parsing, indent=2))