"""
eval/debate_no_memory_alf.py
Setup (iii) — No memory at all.
ALFWorld action-intensive tasks.
Every step only feeds current state + last 5 history.
No ChromaDB, no retrieval, no saving.

Run: CUDA_VISIBLE_DEVICES=3 python eval/debate_no_memory_alf.py
"""

import os, sys, time, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("OPENAI_API_KEY", "dummy-key-not-used")
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import _split

W = 62
MAX_STEPS_PER_AGENT = 15

def line():  print("─" * W)
def box(t):  print(f"\n{'═'*W}\n  {t}\n{'═'*W}")


def run_agent_on_task(agent, task: str, env,
                      prev_responses: list,
                      max_steps: int = MAX_STEPS_PER_AGENT) -> dict:
    """
    Run a single agent on the task.
    Full context given ONCE at start.
    Every step only feeds current state + last 5 history.
    No memory retrieval or saving.
    """
    obs, info   = env.reset()
    current_obs = obs[0]
    done        = False
    success     = False
    total_score = 0.0
    actions     = []
    tokens      = 0
    history     = []  # last N steps only

    # build previous agents context
    prev = ""
    if prev_responses:
        prev = "\nPrevious agents experience:\n"
        for j, r in enumerate(prev_responses):
            prev += f"\nAgent {j+1} ({r['agent_id']}):\n"
            prev += f"  Actions: {r['actions']}\n"
            prev += f"  Result: {r['final_obs'][:80]}\n"

    valid_commands = list(info['admissible_commands'])[0]

    # ── ONCE — full context at start ──────────────────────────
    initial_context = (
        f"You are {agent.agent_id}.\n\n"
        f"Task: {task}\n\n"
        + (f"Previous agents experience:\n{prev}\n\n" if prev else "")
        + f"Solve the task step by step. "
        f"At each step choose ONE action from valid commands.\n"
    )

    for step in range(max_steps):

        # ── EVERY step — only current state + last 5 history ──
        recent = "\n".join(history[-5:]) if history else ""

        prompt = (
            initial_context
            + (f"\n{recent}\n" if recent else "\n")
            + f"Observation: {current_obs}\n"
            + f"Valid commands:\n{chr(10).join(valid_commands[:20])}\n"
            + f"Action:"
        )

        raw    = agent._generate(prompt, max_new_tokens=50)
        tokens += len(agent._tok.encode(raw))

        _, action = _split(raw)

        # fuzzy match ignoring numbers
        action = action.strip().lower()
        action_words = set(w for w in action.split() if not w.isdigit())
        best_score = 0
        matched = None

        for cmd in valid_commands:
            cmd_words = set(w for w in cmd.lower().split() if not w.isdigit())
            overlap = len(action_words & cmd_words)
            if overlap > best_score:
                best_score = overlap
                matched = cmd

        if not matched or best_score == 0:
            matched = valid_commands[0]

        # take action
        new_obs, scores, dones, new_info = env.step([matched])
        current_obs = new_obs[0]
        total_score = scores[0]
        done        = dones[0]
        info        = new_info
        valid_commands = list(new_info['admissible_commands'])[0]

        actions.append(matched)
        history.append(f"Action: {matched}\nObservation: {current_obs[:80]}")

        if done:
            success = total_score > 0
            break

    return {
        "agent_id":  agent.agent_id,
        "actions":   actions,
        "steps":     len(actions),
        "final_obs": current_obs,
        "success":   success,
        "score":     total_score,
        "tokens":    tokens,
        "done":      done,
    }


def run_debate_no_memory_alf(agents: list, env, query_idx: int = 0) -> dict:
    n         = len(agents)
    final_idx = query_idx % n
    rotated   = agents[final_idx+1:] + agents[:final_idx+1]

    obs, info   = env.reset()
    initial_obs = obs[0]
    task        = ""
    for line_ in initial_obs.split('\n'):
        if 'your task is to:' in line_.lower():
            task = line_.strip()
            break

    responses    = []
    token_counts = []
    round_times  = []
    success      = False
    final_score  = 0.0

    end_to_end_start = time.time()

    for i, agent in enumerate(rotated):
        round_start = time.time()

        result = run_agent_on_task(
            agent          = agent,
            task           = task,
            env            = env,
            prev_responses = responses,
        )

        round_time = time.time() - round_start
        responses.append(result)
        token_counts.append(result["tokens"])
        round_times.append(round(round_time, 3))

        if result["success"]:
            success     = True
            final_score = result["score"]
            break

    end_to_end = round(time.time() - end_to_end_start, 3)

    return {
        "setup":        "no_memory",
        "task":         task,
        "final_agent":  rotated[-1].agent_id,
        "responses":    responses,
        "success":      success,
        "score":        final_score,
        "token_counts": token_counts,
        "total_tokens": sum(token_counts),
        "round_times":  round_times,
        "end_to_end":   end_to_end,
    }


if __name__ == "__main__":
    from agents.specialized_agents import AlphaAgent, BetaAgent, GammaAgent, DeltaAgent, EpsilonAgent
    from eval.load_alfworld import load_alfworld_env

    box("DEBATE — NO MEMORY | 5 Agents | ALFWorld")

    print("\n  Loading agents...")
    alpha   = AlphaAgent()
    beta    = BetaAgent()
    gamma   = GammaAgent()
    delta   = DeltaAgent()
    epsilon = EpsilonAgent()
    alpha._load(); beta._load(); gamma._load(); delta._load(); epsilon._load()
    agents = [alpha, beta, gamma, delta, epsilon]
    print("  Agents ready.")

    N   = 50
    env = load_alfworld_env(n=N)

    success_count     = 0
    total_tokens_list = []
    end_to_end_list   = []
    round_times_list  = [[] for _ in range(len(agents))]

    for i in range(N):
        box(f"TASK {i+1}/{N}")
        result = run_debate_no_memory_alf(agents, env, query_idx=i)

        print(f"  Task        : {result['task']}")
        print(f"  Final agent : {result['final_agent']}")
        print(f"\n  [ Per Agent ]")
        line()
        for j, r in enumerate(result["responses"]):
            print(f"  Agent {j+1} ({r['agent_id']})")
            print(f"    steps   : {r['steps']}")
            print(f"    actions : {r['actions']}")
            print(f"    success : {'✓' if r['success'] else '✗'}")
            print(f"    tokens  : {result['token_counts'][j]}")
            print(f"    time    : {result['round_times'][j]}s")

        print(f"\n  [ Summary ]")
        line()
        print(f"  success     : {'✓' if result['success'] else '✗'}")
        print(f"  score       : {result['score']}")
        print(f"  total tokens: {result['total_tokens']}")
        print(f"  end-to-end  : {result['end_to_end']}s")

        if result["success"]:
            success_count += 1

        total_tokens_list.append(result["total_tokens"])
        end_to_end_list.append(result["end_to_end"])
        for j, t in enumerate(result["round_times"]):
            if j < len(round_times_list):
                round_times_list[j].append(t)

    avg_tokens   = round(sum(total_tokens_list) / N, 1)
    avg_e2e      = round(sum(end_to_end_list) / N, 3)
    success_rate = round(success_count / N * 100, 1)

    box("FINAL RESULTS")
    print(f"\n  Setup       : no_memory | 5 agents | ALFWorld | N={N}")
    print(f"\n  1. Time per round (avg across {N} tasks):")
    for j in range(len(agents)):
        if round_times_list[j]:
            avg = round(sum(round_times_list[j]) / len(round_times_list[j]), 3)
            print(f"     Round {j+1} : {avg}s")
    print(f"\n     End-to-end (avg) : {avg_e2e}s")
    print(f"\n  2. Token consumption (avg per task) : {avg_tokens} tokens")
    print(f"\n  3. Overall success rate : {success_count}/{N} = {success_rate}%")