"""
viz_custom 子图测试脚本

模拟 modeling_custom 的输出，测试 viz_custom 子图的完整流程
"""
import os
import json
import pandas as pd
import numpy as np
from ppio_sandbox.code_interpreter import Sandbox
from app.core.config import settings

from app.graph.viz_custom_workflow import build_viz_custom_subgraph
from app.core.viz_custom_subgraph.state import CustomVizState


def prepare_test_data(sandbox: Sandbox):
    """
    准备测试数据：创建一个简单的聚类分析结果
    """
    print("=== 准备测试数据 ===")

    # 创建模拟的聚类数据
    np.random.seed(42)
    n_samples = 100

    # 生成三个聚类中心的数据
    cluster_0 = np.random.randn(30, 2) * 0.5 + np.array([0, 0])
    cluster_1 = np.random.randn(40, 2) * 0.5 + np.array([3, 3])
    cluster_2 = np.random.randn(30, 2) * 0.5 + np.array([6, 0])

    data = np.vstack([cluster_0, cluster_1, cluster_2])
    labels = np.array([0] * 30 + [1] * 40 + [2] * 30)

    # 创建 DataFrame
    df = pd.DataFrame({
        'customer_id': range(100),
        'pca_x': data[:, 0],
        'pca_y': data[:, 1],
        'cluster_label': labels,
        'purchase_amount': np.random.uniform(100, 1000, 100),
        'visit_frequency': np.random.randint(1, 50, 100)
    })

    # 保存为 feather 文件
    df.columns = df.columns.astype(str)
    df.reset_index(drop=True, inplace=True)

    local_path = "temp/clustered_data.feather"
    os.makedirs("temp", exist_ok=True)
    df.to_feather(local_path)

    # 上传到沙箱
    with open(local_path, 'rb') as f:
        file_content = f.read()

    sandbox.files.write("/home/user/clustered_data.feather", file_content)
    print(f"✓ 已上传 clustered_data.feather 到沙箱 ({len(df)} 行)")

    # 创建聚类统计数据（JSON）
    cluster_stats = {
        "n_clusters": 3,
        "silhouette_score": 0.65,
        "cluster_sizes": {
            "cluster_0": 30,
            "cluster_1": 40,
            "cluster_2": 30
        },
        "cluster_centers": {
            "cluster_0": {"pca_x": 0.0, "pca_y": 0.0},
            "cluster_1": {"pca_x": 3.0, "pca_y": 3.0},
            "cluster_2": {"pca_x": 6.0, "pca_y": 0.0}
        }
    }

    local_json_path = "temp/cluster_stats.json"
    with open(local_json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_stats, f, ensure_ascii=False, indent=2)

    # 上传到沙箱
    with open(local_json_path, 'r', encoding='utf-8') as f:
        json_content = f.read()

    sandbox.files.write("/home/user/cluster_stats.json", json_content)
    print(f"✓ 已上传 cluster_stats.json 到沙箱")

    return df


def create_mock_state() -> CustomVizState:
    """
    创建模拟的状态（模拟 modeling_custom 的输出）
    """
    print("\n=== 创建模拟状态 ===")

    # 模拟 modeling_custom 的输出
    modeling_summary = """
```json
[
  {
    "file_name": "clustered_data.feather",
    "columns": [
      { "name": "customer_id", "description": "客户唯一标识" },
      { "name": "pca_x", "description": "PCA 降维后的 X 坐标（新增）" },
      { "name": "pca_y", "description": "PCA 降维后的 Y 坐标（新增）" },
      { "name": "cluster_label", "description": "聚类标签 0/1/2（新增）" },
      { "name": "purchase_amount", "description": "购买金额" },
      { "name": "visit_frequency", "description": "访问频次" }
    ]
  },
  {
    "file_name": "cluster_stats.json",
    "columns": [
      { "name": "n_clusters", "description": "聚类数量" },
      { "name": "silhouette_score", "description": "轮廓系数" },
      { "name": "cluster_sizes", "description": "各聚类的样本数量" },
      { "name": "cluster_centers", "description": "各聚类的中心坐标" }
    ]
  }
]
```
"""

    # 文件元数据（从 modeling_summary 中提取的 JSON）
    file_metadata = [
        {
            "file_name": "clustered_data.feather",
            "columns": [
                {"name": "customer_id", "description": "客户唯一标识"},
                {"name": "pca_x", "description": "PCA 降维后的 X 坐标（新增）"},
                {"name": "pca_y", "description": "PCA 降维后的 Y 坐标（新增）"},
                {"name": "cluster_label", "description": "聚类标签 0/1/2（新增）"},
                {"name": "purchase_amount", "description": "购买金额"},
                {"name": "visit_frequency", "description": "访问频次"}
            ]
        },
        {
            "file_name": "cluster_stats.json",
            "columns": [
                {"name": "n_clusters", "description": "聚类数量"},
                {"name": "silhouette_score", "description": "轮廓系数"},
                {"name": "cluster_sizes", "description": "各聚类的样本数量"},
                {"name": "cluster_centers", "description": "各聚类的中心坐标"}
            ]
        }
    ]

    # 构造初始状态
    state = {
        "messages": [],
        "user_input": "帮我分析客户群体，并进行可视化展示",
        "modeling_summary": modeling_summary,
        "file_metadata": file_metadata,
        "generated_data_files": ["clustered_data.feather", "cluster_stats.json"],
        "error_count": 0
    }

    print("✓ 状态创建完成")
    print(f"  - modeling_summary: {len(modeling_summary)} 字符")
    print(f"  - file_metadata: {len(file_metadata)} 个文件")
    print(f"  - generated_data_files: {state['generated_data_files']}")

    return state


def main():
    """
    主测试流程
    """
    print("=" * 60)
    print("viz_custom 子图测试")
    print("=" * 60)

    # 1. 初始化沙箱
    print("\n=== 初始化沙箱 ===")
    sandbox = sandbox = Sandbox.create(
            settings.PPIO_TEMPLATE,
            api_key=settings.PPIO_API_KEY,
            timeout=3600
    )
    print(f"✓ 沙箱已启动: {sandbox.sandbox_id}")

    try:
        # 2. 准备测试数据
        df = prepare_test_data(sandbox)

        # 3. 创建模拟状态
        state = create_mock_state()

        # 4. 构建 viz_custom 子图
        print("\n=== 构建 viz_custom 子图 ===")
        viz_graph = build_viz_custom_subgraph(sandbox)
        print("✓ 子图构建完成")

        # 5. 运行子图
        print("\n=== 运行 viz_custom 子图 ===")
        print("开始执行...")
        print("-" * 60)

        result = viz_graph.invoke(state)

        print("-" * 60)
        print("✓ 子图执行完成")

        # 6. 输出结果
        print("\n=== 执行结果 ===")

        if "viz_data" in result:
            viz_data = result["viz_data"]
            print(f"✓ 生成了 {len(viz_data.get('echarts', []))} 个图表")

            for idx, chart in enumerate(viz_data.get("echarts", []), 1):
                print(f"\n图表 {idx}:")
                print(f"  类型: {chart.get('type')}")
                print(f"  标题: {chart.get('data', {}).get('title')}")

                # 显示数据摘要
                data = chart.get('data', {})
                if 'xAxis' in data:
                    print(f"  X轴数据点: {len(data['xAxis'])} 个")
                if 'series' in data:
                    print(f"  系列数量: {len(data['series'])} 个")
                    for s_idx, series in enumerate(data['series'], 1):
                        print(f"    系列 {s_idx}: {series.get('name', 'N/A')} ({len(series.get('data', []))} 个数据点)")
        else:
            print("⚠ 未找到 viz_data")

        # 7. 检查本地文件
        print("\n=== 本地产物文件 ===")
        temp_files = [f for f in os.listdir("temp") if f.startswith("viz_")]
        if temp_files:
            print(f"✓ 找到 {len(temp_files)} 个可视化文件:")
            for f in sorted(temp_files):
                file_path = os.path.join("temp", f)
                file_size = os.path.getsize(file_path)
                print(f"  - {f} ({file_size} bytes)")
        else:
            print("⚠ 未找到可视化文件")

        # 8. 显示最终的 viz_data.json
        viz_data_path = "temp/viz_data.json"
        if os.path.exists(viz_data_path):
            print(f"\n=== 最终产物: {viz_data_path} ===")
            with open(viz_data_path, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            print(json.dumps(final_data, ensure_ascii=False, indent=2)[:500] + "...")

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理沙箱
        print("\n=== 清理沙箱 ===")
        try:
            sandbox.kill()
            print("✓ 沙箱已关闭")
        except:
            pass


if __name__ == "__main__":
    main()
