"""agents/specialized_agents.py — 3 agents, each with a distinct model."""

from agents.base_agent import BaseAgent


class AlphaAgent(BaseAgent):
    agent_id = "agent_alpha"
    persona  = "an optimistic analyst who focuses on opportunities and benefits"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"


class BetaAgent(BaseAgent):
    agent_id = "agent_beta"
    persona  = "a critical thinker who identifies risks and counterarguments"
    model_id = "microsoft/phi-2"


class GammaAgent(BaseAgent):
    agent_id = "agent_gamma"
    persona  = "a neutral synthesiser who weighs all perspectives using chain-of-thought reasoning"
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"