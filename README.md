# Night Shift Store Operations Q&A Agent

AI agent for answering convenience store night shift operations questions, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Retail
> **Template ID**: RET-C2-668

## Overview

Grounded operational Q&A for a lone convenience-store night-shift clerk. The clerk asks a
natural-language question — POS/register procedure, age verification for alcohol and tobacco
sales, utility-bill payment (収納代行) or parcel (宅配便) handling, food-safety and 消費期限
checks, equipment troubleshooting, shift handover, or a safety/security/medical/disaster
situation — and the agent retrieves the matching passage from a store-operations-manual and
emergency-protocol knowledge base, then returns a short answer with the source passages cited.

Safety questions are handled differently on purpose. When either the top-ranked passage is an
emergency-protocol entry or an independent English/Japanese keyword scan of the question fires,
the agent stops assembling an answer and returns a **fixed, looked-up escalation instruction**
plus the routing channel (supervisor, 110, or 119). That text is data, not generation, so an
improvised answer to a robbery, medical or disaster question is structurally impossible. When no
passage clears the relevance threshold, the agent says the manual does not cover the question
rather than filling the gap. Every answer carries a standing disclaimer, and a
non-suppressible output gate scans the response for credential-shaped content before it is
returned.

The bundled knowledge base is a small deterministic JSON sample so the pipeline runs and tests
end-to-end out of the box; a real deployment replaces it with its own operations manual and
emergency protocols behind the same node contract. The pipeline is deterministic — keyword
retrieval and rule-based answer assembly, no model call at request time.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and operational documentation
```

See `docs/` for the design specification and test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
