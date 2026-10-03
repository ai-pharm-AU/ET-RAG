"""
MULTI-AGENT ALZHEIMER'S RESEARCH CHATBOT (ET-RAG)
=================================================
Run:  streamlit run INTEGRATED_MULTI_AGENT_COMPLETE.py

Three AI agents answer research questions about the uploaded PDFs:
- Agent 1: Full Context (Gemini 3.8 Flash) - the entire extracted corpus in one prompt
- Agent 2: Cosine RAG (GPT-4o-mini) - multi-query semantic retrieval, top-15 chunks
- Agent 3: ET-RAG (GPT-4o-mini) - evidence-temporal reranking (top-25) + focused hybrid context

The agents, UI, and evaluation helpers live in test.py, so this app, the ablation
study (ablation/run_ablation.py), and the benchmark (benchmark/run_benchmark.py)
all execute a single implementation. The module is cached per server process:
restart Streamlit after editing test.py.
"""

import importlib.util
import sys
from pathlib import Path


def _load_core():
    """Import test.py as 'etrag_core' (the name 'test' is also a stdlib package)."""
    module = sys.modules.get("etrag_core")
    if module is None:
        spec = importlib.util.spec_from_file_location(
            "etrag_core", Path(__file__).resolve().with_name("test.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["etrag_core"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules["etrag_core"]
            raise
    return module


_load_core().main()
