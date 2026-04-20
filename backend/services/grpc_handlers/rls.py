"""Shim: import from utils.grpc_rls to avoid coupling through grpc_handlers."""

from utils.grpc_rls import grpc_admin_rls, grpc_user_rls

__all__ = ["grpc_admin_rls", "grpc_user_rls"]
