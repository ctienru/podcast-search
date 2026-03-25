"""Unit tests for IndexRoutingStrategy."""

import pytest

from src.search.routing import LanguageSplitRoutingStrategy


@pytest.fixture
def strategy() -> LanguageSplitRoutingStrategy:
    return LanguageSplitRoutingStrategy()


def test_routes_zh_tw(strategy):
    assert strategy.get_alias("podcast-episodes-zh-tw") == "episodes-zh-tw"


def test_routes_zh_cn(strategy):
    assert strategy.get_alias("podcast-episodes-zh-cn") == "episodes-zh-cn"


def test_routes_en(strategy):
    assert strategy.get_alias("podcast-episodes-en") == "episodes-en"


def test_raises_for_unknown_target_index(strategy):
    with pytest.raises(ValueError, match="Unknown target_index"):
        strategy.get_alias("podcast-episodes-jp")


def test_raises_for_empty_string(strategy):
    with pytest.raises(ValueError, match="Unknown target_index"):
        strategy.get_alias("")
