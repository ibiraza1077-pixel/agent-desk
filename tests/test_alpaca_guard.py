from types import SimpleNamespace

import pytest

from desk.broker import alpaca_paper as ap


class FakeTrading:
    def __init__(self, base):
        self._base_url = base


def test_live_client_is_refused():
    with pytest.raises(ap.LiveTradingRefused):
        ap.AlpacaPaperBroker("k", "s", trading_client=FakeTrading(ap.LIVE_URL), data_client=object(), news_client=object())


def test_unknown_or_missing_base_url_is_refused():
    for bad in (FakeTrading("https://example.com"), SimpleNamespace(), FakeTrading(None)):
        with pytest.raises(ap.LiveTradingRefused):
            ap.assert_paper(bad)


def test_paper_client_accepted():
    b = ap.AlpacaPaperBroker("k", "s", trading_client=FakeTrading(ap.BaseURL.TRADING_PAPER),
                             data_client=object(), news_client=object())
    assert b.trading._base_url == ap.BaseURL.TRADING_PAPER


def test_constructor_forces_paper_true(monkeypatch):
    seen = {}

    class Rec(FakeTrading):
        def __init__(self, key, secret, **kw):
            seen.update(kw)
            super().__init__(ap.PAPER_URL)

    monkeypatch.setattr(ap, "TradingClient", Rec)
    ap.AlpacaPaperBroker("k", "s", data_client=object(), news_client=object())
    assert seen == {"paper": True}


def test_real_trading_client_is_paper_endpoint():
    # constructing the real alpaca-py client is offline; verifies the attribute we assert on exists
    ap.assert_paper(ap.TradingClient("k", "s", paper=True))
    with pytest.raises(ap.LiveTradingRefused):
        ap.assert_paper(ap.TradingClient("k", "s", paper=False))


def test_paper_mode_requires_keys():
    from desk.config import ConfigError, load_settings
    with pytest.raises(ConfigError) as e:
        load_settings({"DESK_MODE": "paper", "LLM_PROVIDER": "anthropic"}, use_dotenv=False)
    assert "ANTHROPIC_API_KEY" in str(e.value) and "ALPACA_SECRET_KEY" in str(e.value)
    # The default provider is the free local Ollama: only the (free) Alpaca paper keys are needed.
    with pytest.raises(ConfigError) as e:
        load_settings({"DESK_MODE": "paper"}, use_dotenv=False)
    assert "ALPACA_API_KEY" in str(e.value) and "ANTHROPIC_API_KEY" not in str(e.value)
    with pytest.raises(ConfigError):
        load_settings({"WATCHLIST": "SPY,BTC/USD"}, use_dotenv=False)
    with pytest.raises(ConfigError):
        load_settings({"DESK_MODE": "live"}, use_dotenv=False)
