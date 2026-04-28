"""agents/specialized_agents.py — 5 agents, each with a distinct model family."""

from agents.base_agent import BaseAgent


class AlphaAgent(BaseAgent):
    agent_id = "agent_alpha"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"


class BetaAgent(BaseAgent):
    agent_id = "agent_beta"
    model_id = "microsoft/phi-2"


class GammaAgent(BaseAgent):
    agent_id = "agent_gamma"
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class DeltaAgent(BaseAgent):
    agent_id = "agent_delta"
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"


class EpsilonAgent(BaseAgent):
    agent_id = "agent_epsilon"
    model_id = "Qwen/Qwen2.5-7B-Instruct"

