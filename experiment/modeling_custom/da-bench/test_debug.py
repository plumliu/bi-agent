#!/usr/bin/env python3
"""调试脚本 - 获取完整错误信息"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_jsonl, build_initial_state
from src.runner import run_single_question
from ppio_sandbox.code_interpreter import Sandbox
from app.core.config import settings

async def main():
    # 加载第一个问题
    benchmark_path = Path(__file__).parent.parent.parent.parent / "infiagent-DABench"
    questions = load_jsonl(benchmark_path / "da-dev-questions.jsonl")
    q = questions[0]

    file_path = benchmark_path / "da-dev-tables" / q['file_name']

    print(f"测试问题: {q['id']}")
    print(f"文件路径: {file_path}")

    # 创建 sandbox
    sandbox = Sandbox.create(
        settings.PPIO_TEMPLATE,
        api_key=settings.PPIO_API_KEY,
        timeout=3600
    )

    try:
        result = await run_single_question(sandbox, q, file_path)
        print("\n结果:")
        print(f"Success: {result['success']}")
        print(f"Output: {result['output'][:200] if result['output'] else 'None'}")
        if result['error']:
            print(f"\n完整错误:\n{result['error']}")
    finally:
        sandbox.kill()

if __name__ == "__main__":
    asyncio.run(main())
