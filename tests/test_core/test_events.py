import asyncio
import time

import pytest

from bytecli.core.events import Event, EventBus


class TestEvent:
    def test_create_event(self):
        event = EventBus.create_event("test.event", {"key": "value"})
        assert event.type == "test.event"
        assert event.data == {"key": "value"}
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0

    def test_event_default_data(self):
        event = EventBus.create_event("test.empty")
        assert event.data == {}

    def test_event_frozen(self):
        event = EventBus.create_event("test.frozen")
        with pytest.raises(AttributeError):
            event.data = {}  # type: ignore[misc]

    def test_event_empty_type_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Event(type="", timestamp=0.0, data={})

    def test_event_data_must_be_dict(self):
        with pytest.raises(ValueError, match="dict"):
            Event(type="test", timestamp=0.0, data=[])  # type: ignore[arg-type]


class TestEventBus:
    @pytest.mark.asyncio
    async def test_basic_publish_subscribe(self, event_bus: EventBus):
        received: list[Event] = []

        async def listener(event: Event) -> None:
            received.append(event)

        event_bus.subscribe("test.event", listener)
        await event_bus.publish(EventBus.create_event("test.event", {"n": 1}))
        assert len(received) == 1
        assert received[0].data["n"] == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus: EventBus):
        received: list[Event] = []

        async def listener(event: Event) -> None:
            received.append(event)

        event_bus.subscribe("test.event", listener)
        await event_bus.publish(EventBus.create_event("test.event"))
        assert len(received) == 1

        event_bus.unsubscribe("test.event", listener)
        await event_bus.publish(EventBus.create_event("test.event"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus: EventBus):
        received: list[str] = []

        async def listener(event: Event) -> None:
            received.append(event.type)

        event_bus.subscribe("tool.*", listener)
        await event_bus.publish(EventBus.create_event("tool.execution.start"))
        await event_bus.publish(EventBus.create_event("tool.execution.end"))
        await event_bus.publish(EventBus.create_event("provider.request.start"))
        assert received == ["tool.execution.start", "tool.execution.end"]

    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_bus: EventBus):
        order: list[int] = []

        async def listener_high(event: Event) -> None:
            order.append(1)

        async def listener_low(event: Event) -> None:
            order.append(2)

        event_bus.subscribe("test", listener_high, priority=10)
        event_bus.subscribe("test", listener_low, priority=20)
        await event_bus.publish(EventBus.create_event("test"))
        assert order == [1, 2]

    @pytest.mark.asyncio
    async def test_listener_error_does_not_crash_bus(self, event_bus: EventBus):
        received: list[Event] = []

        async def bad_listener(event: Event) -> None:
            raise RuntimeError("oops")

        async def good_listener(event: Event) -> None:
            received.append(event)

        event_bus.subscribe("test", bad_listener, priority=0)
        event_bus.subscribe("test", good_listener, priority=10)
        await event_bus.publish(EventBus.create_event("test"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_topic(self, event_bus: EventBus):
        received: list[int] = []

        async def listener1(event: Event) -> None:
            received.append(1)

        async def listener2(event: Event) -> None:
            received.append(2)

        event_bus.subscribe("test", listener1)
        event_bus.subscribe("test", listener2)
        await event_bus.publish(EventBus.create_event("test"))
        assert 1 in received
        assert 2 in received

    @pytest.mark.asyncio
    async def test_no_effect_on_other_topics(self, event_bus: EventBus):
        received: list[str] = []

        async def listener(event: Event) -> None:
            received.append(event.type)

        event_bus.subscribe("topic.a", listener)
        await event_bus.publish(EventBus.create_event("topic.b"))
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_auto_log_filtering(self):
        filtered_bus = EventBus(auto_log=True, event_filter={"important.event"})
        normal_bus = EventBus(auto_log=True, event_filter=None)

        assert filtered_bus._should_log("important.event") is True
        assert filtered_bus._should_log("noisy.event") is False
        assert normal_bus._should_log("anything") is True

    @pytest.mark.asyncio
    async def test_sanitize_sensitive_data(self, event_bus: EventBus):
        data = event_bus._sanitize_for_log({"api_key": "sk-1234567890", "normal": "value"})
        assert data["api_key"] == "***"
        assert data["normal"] == "value"

    @pytest.mark.asyncio
    async def test_sanitize_large_data(self, event_bus: EventBus):
        data = event_bus._sanitize_for_log({"big": "x" * 1000})
        assert len(data["big"]) == 503

    @pytest.mark.asyncio
    async def test_listener_count(self, event_bus: EventBus):
        async def listener(event: Event) -> None:
            pass

        assert event_bus.listener_count() == 0
        event_bus.subscribe("test", listener)
        assert event_bus.listener_count() == 1
        event_bus.unsubscribe("test", listener)
        assert event_bus.listener_count() == 0
