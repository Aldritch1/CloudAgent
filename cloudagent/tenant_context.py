import contextvars

tenant_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "tenant_id", default=""
)


def get_tenant_id() -> str:
    return tenant_ctx.get()


def set_tenant_id(tenant_id: str) -> None:
    tenant_ctx.set(tenant_id)
