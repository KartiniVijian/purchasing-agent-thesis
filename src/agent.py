"""agent.py - simplified ReAct-style purchasing agent."""
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from decision_function import Goal, Decision, evaluate_product


@dataclass
class AgentStepLog:
    listing_title: str
    source_file: str
    decision: Decision


class PurchasingAgent:
    def __init__(self, model_name: str = "llama3.1:8b", llm=None, temperature: float = 0.0):
        self.llm = llm if llm is not None else ChatOllama(model=model_name, temperature=temperature)
        self.trace = []

    def evaluate_listing(self, goal: Goal, listing: dict) -> Decision:
        decision = evaluate_product(self.llm, goal, listing)
        self.trace.append(
            AgentStepLog(
                listing_title=listing.get("title", "UNKNOWN"),
                source_file=listing.get("_source_file", "unknown"),
                decision=decision,
            )
        )
        return decision

    def evaluate_all(self, goal: Goal, listings: list) -> list:
        return [self.evaluate_listing(goal, listing) for listing in listings]

    def print_trace(self) -> None:
        for step in self.trace:
            status = "ACCEPTED" if step.decision.accepted else "REJECTED"
            print(f"[{status}] {step.listing_title}  ({step.source_file})")
            print(f"    reason: {step.decision.reason}")
