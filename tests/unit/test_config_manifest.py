# RET-C2-668 — Unit Tests: manifest / config consistency
#
# Two files, two jobs, and these tests pin the split:
#   config/agent.yaml   the static manifest — identity, entry point, trust
#                       level, compile-time `requires` gates. Flat: every key
#                       at root level, no `agent:` nesting.
#   config/config.yaml  runtime values — max_retry / timeout_s and the
#                       `retrieval` / `llm` tuning blocks.
#
# A reader pointed at the wrong file does not fail; it finds no such key, falls
# back to its own default, and every declared value goes dead while the tests
# stay green. So these tests assert the READERS land on live values, not just
# that the files parse.
#
# Mirrors docs/03_test_spec.md CFG-01..CFG-09.
# Deterministic — no LLM, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import ConvenienceStoreNightShiftOpsAgent, NightShiftOpsGraphNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.services.runtime_config import agent_config, llm_config, retrieval_config

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_CONFIG = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

_ESCALATION_CATEGORIES = {"emergency_security", "emergency_medical", "emergency_disaster"}


class TestManifestIdentity:
    def test_cfg_01_template_id_is_at_root(self):
        # Flat schema: the registry reads every key at root level.
        assert _MANIFEST["id"] == "RET-C2-668"
        assert _MANIFEST["name"] == ConvenienceStoreNightShiftOpsAgent().name
        assert _MANIFEST["namespace"] == "ret"
        assert "agent" not in _MANIFEST, "the manifest must not carry a nested agent: block"

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # Class-name contract: manifest class == graph.py class == server import.
        assert _MANIFEST["class"] == (f"src.graph.graph.{ConvenienceStoreNightShiftOpsAgent.__name__}")

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "RET"
        assert _MANIFEST["base_type"] == "RAGAgent"

    def test_cfg_09_compile_time_requires_are_empty(self):
        # The pipeline is deterministic and calls no external service: it
        # requires no secret and constructs no optional client. Declaring
        # either would make the agent fail to compile at deploy time waiting on
        # something nothing provisions.
        assert _MANIFEST["generation_mode"] == "deterministic"
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_cfg_05_max_retry_within_framework_ceiling(self):
        max_retry = _CONFIG["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # AgentBaseGraph MAX_RETRY_CEILING

    def test_hitl_is_not_enabled(self):
        # PB-7 auto-waiver contract: this template declares no HITL.
        assert (_CONFIG.get("hitl") or {}).get("enabled", False) is False


class TestRuntimeConfigIsLive:
    """The declared runtime values reach the code that consumes them."""

    def test_cfg_06_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror the declared block — a drift silently
        # changes tuning for anyone running without state seeding.
        retrieval = _CONFIG["retrieval"]
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults

        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_07_parent_config_forwards_the_runtime_blocks(self):
        cfg = NightShiftOpsGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == _CONFIG["retrieval"]
        assert cfg["configurable"]["llm"] == _CONFIG["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty block"

    def test_runtime_loader_reads_config_yaml_not_the_manifest(self):
        # The regression this guards: readers left pointing at the manifest
        # return {} for these keys and degrade silently to their own defaults.
        assert retrieval_config() == _CONFIG["retrieval"]
        assert llm_config() == _CONFIG["llm"]
        assert "retrieval" not in _MANIFEST
        assert "llm" not in _MANIFEST

    def test_agent_config_carries_the_graph_level_values(self):
        cfg = agent_config()
        assert cfg["max_retry"] == _CONFIG["max_retry"]
        assert cfg["timeout_s"] == _CONFIG["timeout_s"]

    def test_declared_max_retry_reaches_the_compiled_graph(self):
        # The entry point constructs the agent with agent_config(); prove the
        # framework backbone reads the declared value rather than its own
        # default.
        agent = ConvenienceStoreNightShiftOpsAgent(config=agent_config())
        agent.compile()
        assert agent.config["max_retry"] == _CONFIG["max_retry"]


class TestSeededKnowledgeBase:
    def _entries(self):
        return json.loads((_ROOT / _CONFIG["retrieval"]["kb_path"]).read_text(encoding="utf-8"))

    def test_cfg_08_kb_is_a_well_formed_entry_list(self):
        entries = self._entries()
        assert isinstance(entries, list)
        assert len(entries) >= 5, "seeded knowledge base must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        ids = [e["id"] for e in self._entries()]
        assert len(ids) == len(set(ids))

    def test_kb_carries_all_three_escalation_categories(self):
        # The safety boundary's first signal depends on these three categories
        # actually existing in the seed — a knowledge-base edit that drops one
        # silently weakens it (the keyword net, the second signal, still
        # applies independently).
        categories = {e["category"] for e in self._entries()}
        assert _ESCALATION_CATEGORIES <= categories

    def test_kb_categories_are_inert_identifiers(self):
        # Callers select a category through the caller contract, which locks the
        # value to an inert alphabet. A seeded category outside that alphabet
        # would be unselectable.
        from src.services.caller_contract import require_inert_identifier

        for entry in self._entries():
            assert require_inert_identifier(entry["category"], "category") == entry["category"]
