from desk.broker.sim import SimBroker
from desk.desk import Desk
from desk.agents.scripted import ScriptedTeam


class CountingLLMTeam(ScriptedTeam):
    uses_llm = True

    def __init__(self, s):
        super().__init__(s)
        self.calls = 0

    def analyst(self, facts):
        self.calls += 1
        return super().analyst(facts)

    def review(self, facts):
        self.calls += 1
        return super().review(facts)


def test_cycle_skips_llm_when_budget_spent(settings, store):
    settings.demo_agents = "llm"  # paid Claude agents (fixture provider is anthropic) on the sim market
    assert not settings.llm_is_free
    team = CountingLLMTeam(settings)
    desk = Desk(settings, store, SimBroker(settings.watchlist, 1), team)
    store.add_usage("Chief", "claude-opus-5-5", 1, 1, settings.daily_llm_budget_usd)  # exactly at budget
    desk.run_cycle()
    assert team.calls == 0
    texts = [f["text"] for f in store.get_feed()]
    assert any("LLM budget reached" in t for t in texts)
    assert store.get_equity()  # the cycle still snapshots equity


def test_cycle_uses_llm_under_budget(settings, store):
    team = CountingLLMTeam(settings)
    desk = Desk(settings, store, SimBroker(settings.watchlist, 1), team)
    desk.run_cycle()
    assert team.calls > 0


def test_free_provider_is_never_budget_gated(settings, store):
    settings.demo_agents, settings.llm_provider, settings.daily_llm_budget_usd = "llm", "ollama", 0.0
    team = CountingLLMTeam(settings)
    desk = Desk(settings, store, SimBroker(settings.watchlist, 1), team)
    store.add_usage("Chief", "claude-opus-5-5", 1, 1, 3.0)  # earlier paid spend today
    desk.run_cycle()
    assert team.calls > 0
