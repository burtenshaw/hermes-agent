from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        instance.client = MagicMock()
        instance._cached_system_prompt = "You are helpful."
        instance._use_prompt_caching = False
        instance.tool_delay = 0
        instance.compression_enabled = False
        instance.save_trajectories = False
        return instance


def test_build_api_kwargs_passes_llama_cpp_parallel_flag(agent):
    agent.provider = "llama-cpp"
    agent._llama_cpp_parallel_tool_calls = False

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])

    assert "tools" in kwargs
    assert kwargs["parallel_tool_calls"] is False


def test_llama_cpp_tool_turn_uses_parser_fallback_and_generated_ids(agent):
    agent.provider = "llama-cpp"
    agent._llama_cpp_parser_chain = ["qwen3_coder"]
    agent._llama_cpp_streaming_tool_calls = True
    agent.stream_delta_callback = lambda _delta: None

    resp1 = _mock_response(
        content='<tool_call><function=web_search><parameter=q>"test"</parameter></function></tool_call>',
        finish_reason="tool",
    )
    resp2 = _mock_response(content="Done searching", finish_reason="stop")
    parsed_tool_call = SimpleNamespace(
        id="",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"q":"test"}'),
    )
    parser = SimpleNamespace(
        parse=lambda text: (
            ("", [parsed_tool_call])
            if "<function=web_search>" in text
            else (text, [])
        )
    )

    with (
        patch("environments.tool_call_parsers.get_parser", return_value=parser),
        patch.object(agent, "_interruptible_api_call") as mock_nonstream,
        patch.object(agent, "_interruptible_streaming_api_call", side_effect=[resp1, resp2]) as mock_stream,
        patch("run_agent.handle_function_call", return_value="search result"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search something")

    assert result["final_response"] == "Done searching"
    assert mock_stream.call_count == 2
    mock_nonstream.assert_not_called()

    assistant_tool_turns = [
        message
        for message in result["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(assistant_tool_turns) == 1
    tool_call = assistant_tool_turns[0]["tool_calls"][0]
    assert tool_call["function"]["name"] == "web_search"
    assert tool_call["id"].startswith("call_")
