from typing import Optional

from fastapi import Depends, Header

from cloudagent.auth import get_current_user
from cloudagent.tenant_context import get_tenant_id, set_tenant_id


async def tenant_dependency(
    user_id: str = Depends(get_current_user),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> str:
    from cloudagent.config import settings

    tenant_id = get_tenant_id() or x_tenant_id or settings.default_tenant_id
    set_tenant_id(tenant_id)
    return tenant_id
