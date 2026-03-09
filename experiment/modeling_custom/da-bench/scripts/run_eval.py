#!/usr/bin/env python3
"""DABench 评估脚本 - 生成 responses.jsonl"""
import sys
import os
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录和 da-bench 目录到 Python 路径
project_root = Path(__file__).parent.parent.parent.parent.parent
da_bench_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(da_bench_dir))

from src.utils import load_jsonl
from src.sandbox_manager import SandboxPool
from src.runner import run_single_question


async def main():
    parser = argparse.ArgumentParser(description='DABench评估脚本')
    parser.add_argument('--benchmark-dir', type=str,
                        default='../../../../infiagent-DABench',
                        help='Benchmark数据目录')
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出文件（默认：results/responses_TIMESTAMP.jsonl）')
    parser.add_argument('--limit', type=int, default=None,
                        help='限制评估的问题数量（用于测试）')
    parser.add_argument('--max-concurrent', type=int, default=45,
                        help='最大并发sandbox数量')

    args = parser.parse_args()

    # 加载 benchmark 数据（使用绝对路径）
    script_dir = Path(__file__).parent
    benchmark_path = (script_dir / args.benchmark_dir).resolve()

    if not benchmark_path.exists():
        print(f"错误: Benchmark 目录不存在: {benchmark_path}")
        return

    questions = load_jsonl(benchmark_path / "da-dev-questions.jsonl")

    if args.limit:
        questions = questions[:args.limit]

    # 设置输出文件
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"results/responses_{timestamp}.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"开始评估 {len(questions)} 个问题...")
    print(f"结果将保存至: {output_path}")
    print("=" * 80)

    # 创建 sandbox 池
    pool = SandboxPool(max_concurrent=args.max_concurrent)

    # 并发运行所有问题
    tasks = []
    for idx, q in enumerate(questions):
        file_path = benchmark_path / "da-dev-tables" / q['file_name']
        task = pool.run_with_sandbox(run_single_question, q, file_path)
        tasks.append((idx, q, task))

    # 并发执行所有任务
    print(f"启动 {len(tasks)} 个并发任务（最多同时 {pool.max_concurrent} 个）...")
    task_results = await asyncio.gather(*[task for _, _, task in tasks])

    # 保存结果为 JSONL 格式（官方格式）
    with open(output_path, 'w', encoding='utf-8') as f:
        for (idx, q, _), result in zip(tasks, task_results):
            q_id = q['id']
            output = result['output']
            error = result['error']

            # 官方格式：{"id": 0, "response": "@mean_fare[34.65]"}
            response_entry = {
                "id": q_id,
                "response": output if result['success'] else ""
            }
            f.write(json.dumps(response_entry, ensure_ascii=False) + '\n')

            # 打印进度
            status = "✓" if result['success'] else "✗"
            print(f"[{idx+1}/{len(questions)}] ID={q_id} {status}")
            if error:
                print(f"  错误: {error[:200]}...")

    print("\n" + "=" * 80)
    print(f"✓ 所有问题已完成，结果已保存至: {output_path}")
    print("\n使用官方评估脚本计算指标:")
    print(f"  python official_implementation\\(reference\\)/eval_closed_form.py \\")
    print(f"    --questions_file_path {benchmark_path}/da-dev-questions.jsonl \\")
    print(f"    --labels_file_path {benchmark_path}/da-dev-labels.jsonl \\")
    print(f"    --responses_file_path {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
