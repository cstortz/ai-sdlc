"""
agents/intake/interviewer.py — Clarifying-question interface.

Abstracts *how* questions are asked so the IntakeAgent doesn't care
whether it's talking to a CLI or a Redmine ticket thread.

Active implementation:  CliInterviewer  (Phase 1)
Future implementation:  RedmineInterviewer (stub — Phase 2+)

To add a new interview mode:
  1. Subclass BaseInterviewer
  2. Implement ask() and confirm()
  3. Add the mode to InterviewMode and create_interviewer()
"""
from __future__ import annotations

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class InterviewMode(str, Enum):
    CLI     = "cli"
    REDMINE = "redmine"


class BaseInterviewer(ABC):
    """Abstract interface for asking clarifying questions."""

    @abstractmethod
    async def ask(self, question: str, context: str = "") -> str:
        """
        Present a question and return the human's answer as a string.
        `context` is optional background shown before the question.
        """

    @abstractmethod
    async def confirm(self, prompt: str) -> bool:
        """Yes/no confirmation. Returns True for yes."""

    @abstractmethod
    async def show(self, message: str) -> None:
        """Display a message to the human (no response expected)."""


# ---------------------------------------------------------------------------
# CLI implementation (Phase 1 — active)
# ---------------------------------------------------------------------------

class CliInterviewer(BaseInterviewer):
    """
    Asks questions interactively in the terminal.

    Runs input() in a thread executor so it doesn't block the event loop.
    """

    SEPARATOR = "─" * 60

    async def ask(self, question: str, context: str = "") -> str:
        if context:
            print(f"\n{context}")
        print(f"\n{self.SEPARATOR}")
        print(f"  {question}")
        print(f"{self.SEPARATOR}")
        answer = await self._input("  › ")
        return answer.strip()

    async def confirm(self, prompt: str) -> bool:
        print(f"\n{self.SEPARATOR}")
        print(f"  {prompt} [y/N]")
        print(f"{self.SEPARATOR}")
        answer = await self._input("  › ")
        return answer.strip().lower() in ("y", "yes")

    async def show(self, message: str) -> None:
        print(f"\n{message}")

    @staticmethod
    async def _input(prompt: str) -> str:
        """Non-blocking input() via thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: input(prompt))


# ---------------------------------------------------------------------------
# Redmine implementation (stub — Phase 2+)
# ---------------------------------------------------------------------------

class RedmineInterviewer(BaseInterviewer):
    """
    Asks clarifying questions by posting comments on a Redmine ticket
    and polling for a reply.

    NOT YET IMPLEMENTED — raises NotImplementedError.
    Wire up when moving to Phase 2 (gate_points autonomy).

    To implement:
      - POST /issues/{id}/notes.json  with the question text
      - Poll GET /issues/{id}/journals.json  until a new comment appears
        that isn't from the agent user
      - Return the comment body as the answer
    """

    def __init__(self, issue_id: int, redmine_url: str, api_key: str, agent_user_id: int):
        self.issue_id = issue_id
        self.redmine_url = redmine_url
        self.api_key = api_key
        self.agent_user_id = agent_user_id

    async def ask(self, question: str, context: str = "") -> str:
        raise NotImplementedError(
            "RedmineInterviewer is not yet implemented. "
            "Use InterviewMode.CLI for Phase 1."
        )

    async def confirm(self, prompt: str) -> bool:
        raise NotImplementedError("RedmineInterviewer.confirm() not yet implemented.")

    async def show(self, message: str) -> None:
        raise NotImplementedError("RedmineInterviewer.show() not yet implemented.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_interviewer(
    mode: InterviewMode | str = InterviewMode.CLI,
    **kwargs,
) -> BaseInterviewer:
    """
    Factory function — returns the right interviewer for the given mode.

    Examples:
        create_interviewer("cli")
        create_interviewer("redmine", issue_id=42, redmine_url=..., api_key=..., agent_user_id=1)
    """
    mode = InterviewMode(mode)
    if mode == InterviewMode.CLI:
        return CliInterviewer()
    if mode == InterviewMode.REDMINE:
        return RedmineInterviewer(**kwargs)
    raise ValueError(f"Unknown interview mode: {mode}")
