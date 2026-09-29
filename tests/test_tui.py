"""Headless tests for the terminal UI (no network, no model inference).

These tests run fully offline: the app falls back to the configured default
model when the local Ollama server is unreachable, and composing/running the
UI does not require model inference.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_rag.tui import ChatApp
from textual.containers import VerticalScroll
from textual.widgets import Input, Select, Static


def _run(coro):
    return asyncio.run(coro)


def test_chat_app_composes_with_settings():
    async def main():
        app = ChatApp()
        async with app.run_test():
            model_select = app.query_one("#model-select", Select)
            filter_select = app.query_one("#filter-select", Select)
            topk = app.query_one("#topk", Input)
            qinput = app.query_one("#qinput", Input)
            assert model_select.value  # at least the default model
            assert filter_select.value is not None
            assert int(topk.value) > 0
            assert qinput is not None

    _run(main())


def test_help_command_renders_info_bubble():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/help"
            await qinput.action_submit()
            await pilot.pause()
            assert list(app.query(".bubble.info"))

    _run(main())


def test_clear_command_resets_chat():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/help"
            await qinput.action_submit()
            await pilot.pause()
            qinput.value = "/clear"
            await qinput.action_submit()
            await pilot.pause()
            chat = app.query_one("#chat", VerticalScroll)
            assert len(list(chat.children)) == 1

    _run(main())


def test_stats_command_updates_statsbar():
    async def main():
        app = ChatApp()
        async with app.run_test() as pilot:
            qinput = app.query_one("#qinput", Input)
            qinput.value = "/stats"
            await qinput.action_submit()
            await pilot.pause()
            statsbar = app.query_one("#statsbar", Static)
            assert "chunks:" in str(statsbar.render())

    _run(main())
