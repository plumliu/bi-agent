"""Sandbox 池管理模块"""
import asyncio
import time
from ppio_sandbox.code_interpreter import Sandbox
from app.core.config import settings


class SandboxPool:
    """管理 sandbox 实例池，支持并发控制"""

    def __init__(self, max_concurrent: int = 45):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_sandbox(self, task_func, *args):
        """
        在独立 sandbox 中运行任务

        Args:
            task_func: 异步任务函数，第一个参数必须是 sandbox
            *args: 传递给 task_func 的其他参数

        Returns:
            任务函数的返回值
        """
        async with self.semaphore:
            sandbox = None
            max_retries = 3
            retry_delay = 5

            for attempt in range(max_retries):
                try:
                    # 创建 sandbox（完全参考 main.py:119-123）
                    sandbox = Sandbox.create(
                        settings.PPIO_TEMPLATE,
                        api_key=settings.PPIO_API_KEY,
                        timeout=3600  # 1小时超时
                    )
                    print(f"  ✓ Sandbox 创建成功: {sandbox.sandbox_id}")

                    # 执行任务
                    result = await task_func(sandbox, *args)
                    return result

                except Exception as e:
                    error_msg = str(e)
                    if "Connection reset" in error_msg or "ConnectError" in error_msg:
                        if attempt < max_retries - 1:
                            print(f"  ⚠ Sandbox 创建失败 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}")
                            print(f"  ⏳ 等待 {retry_delay} 秒后重试...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # 指数退避
                            continue
                        else:
                            print(f"  ✗ Sandbox 创建失败，已达最大重试次数: {error_msg[:100]}")
                            raise
                    else:
                        # 非网络错误，直接抛出
                        raise

                finally:
                    # 清理 sandbox（完全参考 main.py:268-273）
                    if sandbox:
                        try:
                            sandbox.kill()
                            print(f"  ✓ Sandbox 已清理: {sandbox.sandbox_id}")
                        except Exception as e:
                            print(f"  ✗ Sandbox 清理失败: {e}")
