"""
scripts/load_memories.py
Run: python scripts/load_memories.py memory/test_dataset.py
"""

import os, sys, warnings, logging, importlib.util
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, ".")

from memory.mem0_store import save

path   = sys.argv[1] if len(sys.argv) > 1 else "memory/test_dataset.py"
spec   = importlib.util.spec_from_file_location("dataset", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

memories = next(
    v for v in vars(module).values()
    if isinstance(v, list) and v and isinstance(v[0], dict)
)

for m in memories:
    save(m["query"], m["reasoning"], m["answer"], m["agent_id"])

print(f"{len(memories)} memories inserted from {path}")