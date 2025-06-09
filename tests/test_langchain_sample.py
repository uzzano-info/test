import builtins
from types import SimpleNamespace
import sys
import importlib
import types
import pytest


def load_sample(monkeypatch):
    """Import sample module with stubbed dependencies."""
    stub_openai = types.ModuleType("langchain_openai")

    class DummyLLM:
        instances = []

        def __init__(self, *args, **kwargs):
            self.model = kwargs.get("model")
            DummyLLM.instances.append(self)

        def __ror__(self, other):
            return self

        def invoke(self, params):
            return SimpleNamespace(content=f"echo: {params['question']}")

    stub_openai.ChatOpenAI = DummyLLM
    monkeypatch.setitem(sys.modules, "langchain_openai", stub_openai)

    stub_langchain = types.ModuleType("langchain")
    prompts = types.ModuleType("prompts")
    class DummyPrompt:
        def __init__(self, msgs):
            self.msgs = msgs
        def __or__(self, other):
            return other
        @classmethod
        def from_messages(cls, msgs):
            return cls(msgs)
    prompts.ChatPromptTemplate = DummyPrompt
    stub_langchain.prompts = prompts
    monkeypatch.setitem(sys.modules, "langchain", stub_langchain)
    monkeypatch.setitem(sys.modules, "langchain.prompts", prompts)

    module_name = "examples.langchain_sample"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name), DummyLLM


def test_missing_api_key(monkeypatch):
    sample, DummyLLM = load_sample(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        sample.main()
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_custom_question(monkeypatch):
    sample, DummyLLM = load_sample(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # capture output
    monkeypatch.setattr(builtins, "print", lambda msg: setattr(test_custom_question, "output", msg))
    monkeypatch.setattr(sys, "argv", ["langchain_sample.py", "hello"])

    sample.main()
    assert getattr(test_custom_question, "output", "") == "echo: hello"


def test_custom_model(monkeypatch):
    sample, DummyLLM = load_sample(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    monkeypatch.setattr(sys, "argv", ["langchain_sample.py", "hello", "--model", "gpt-x"])
    sample.main()
    assert DummyLLM.instances[-1].model == "gpt-x"
