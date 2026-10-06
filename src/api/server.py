"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, AgentGateway calls agent.invoke() directly.

import os
import secrets
from typing import Any, Dict, Optional, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import ConvenienceStoreNightShiftOpsAgent
from src.services.runtime_config import agent_config

app = FastAPI(title="Agent")

# The declared runtime values (config/config.yaml) are passed in at
# construction. Without this the graph runs on the framework's built-in
# defaults and every value in that file is dead.
agent = ConvenienceStoreNightShiftOpsAgent(config=agent_config())
agent.compile()
# Replace namespace/agent_name to match the agent's manifest values.
agent.provision_secrets(secrets_factory(namespace="ret-c2-668", agent_name="ConvenienceStoreNightShiftOpsAgent"))


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured caller parameters (category / top_k / score_threshold /
    # channel). Validated by PreProcessNode against explicit bounds - this
    # adapter passes the payload through untouched and never interprets it.
    input_context: Optional[Dict[str, Any]] = Field(default=None)


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> Dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone caller auth: when INVOKE_AUTH_TOKEN is set on the server
    # environment, callers that no upstream middleware vouched for (still
    # ANONYMOUS) must present it as a Bearer token and run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary — a deployment-level caller
    # credential, not an agent secret, so ctx.secrets does not apply: no
    # InvocationContext exists before auth.
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL
    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        # cast: the framework wheel ships no type information, so invoke()
        # resolves to Any here.
        return cast(
            Dict[str, Any],
            agent.invoke(req.input, ctx=ctx, input_context=req.input_context or {}),
        )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "agent": "ConvenienceStoreNightShiftOpsAgent"}
