from cloudagent.hitl import HITLManager


def test_is_sensitive():
    hitl = HITLManager()
    assert hitl.is_sensitive("workflow", {"action": "refund"}) is True
    assert hitl.is_sensitive("workflow", {"action": "cancel"}) is True
    assert hitl.is_sensitive("workflow", {"action": "delete"}) is True
    assert hitl.is_sensitive("workflow", {"action": "query"}) is False


def test_build_confirmation_message():
    hitl = HITLManager()
    msg = hitl.build_confirmation_message("refund", {"order_id": "123"})
    assert "refund" in msg
    assert "确认" in msg


def test_is_confirm():
    hitl = HITLManager()
    assert hitl.is_confirm("确认") is True
    assert hitl.is_confirm("是的") is True
    assert hitl.is_confirm("confirm") is True
    assert hitl.is_confirm("no") is False


def test_is_reject():
    hitl = HITLManager()
    assert hitl.is_reject("取消") is True
    assert hitl.is_reject("reject") is True
    assert hitl.is_reject("确认") is False
