#!/usr/bin/env python3
"""计算 DABench 评估指标（基于官方实现）"""
import sys
import json
import argparse
import re
from pathlib import Path

# 添加 da-bench 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_jsonl


def extract_format(input_string):
    """提取 @key[value] 格式的答案"""
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers


def is_equal(response, label):
    """判断两个答案是否相等（支持数值比较，容差 1e-6）"""
    if response == label:
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except:
        return False


def evaluate_responses(labels, responses):
    """评估所有响应"""
    results = []
    for label in labels:
        label_id = label["id"]
        label_answers = {ans[0]: ans[1] for ans in label["common_answers"]}

        corresponding_response = next(
            (resp for resp in responses if "id" in resp.keys() and resp.get('response') and resp["id"] == label_id),
            None
        )

        if corresponding_response:
            answer_names, answers = extract_format(corresponding_response["response"])
            extracted_answers = dict(zip(answer_names, answers))
            correct_answers = {
                ans_name: is_equal(extracted_answers.get(ans_name), label_answers[ans_name])
                for ans_name in label_answers.keys()
            }
            result = {
                "id": label_id,
                "label_answers": label_answers,
                "predicted_answers": extracted_answers,
                "correctness": correct_answers
            }
            results.append(result)
    return results


def read_concepts_from_questions(questions):
    """从问题文件中读取 concept 信息"""
    concepts_data = {}
    for question in questions:
        question_id = question["id"]
        concepts = question.get("concepts", [])
        concepts_data[question_id] = concepts
    return concepts_data


def analyze_concepts_accuracy(results, concepts_data):
    """按 concept 分析准确率"""
    concept_accuracy = {}
    for result in results:
        question_id = result["id"]
        if question_id in concepts_data:
            for concept in concepts_data[question_id]:
                if concept not in concept_accuracy:
                    concept_accuracy[concept] = {"total": 0, "correct": 0}
                concept_accuracy[concept]["total"] += 1
                if 'correctness' in result and all(result['correctness'].values()):
                    concept_accuracy[concept]["correct"] += 1
    return {concept: round(acc["correct"] / acc["total"], 4) for concept, acc in concept_accuracy.items()}


def analyze_concepts_count_accuracy(results, concepts_data):
    """按 concept 数量分析准确率"""
    concept_count_accuracy = {}
    two_or_more_concepts_accuracy = {"total": 0, "correct": 0}

    for result in results:
        question_id = result["id"]
        if question_id in concepts_data:
            concept_count = len(concepts_data[question_id])
            if concept_count not in concept_count_accuracy:
                concept_count_accuracy[concept_count] = {"total": 0, "correct": 0}
            concept_count_accuracy[concept_count]["total"] += 1

            if 'correctness' in result and all(result['correctness'].values()):
                concept_count_accuracy[concept_count]["correct"] += 1

            if concept_count >= 2:
                two_or_more_concepts_accuracy["total"] += 1
                if 'correctness' in result and all(result['correctness'].values()):
                    two_or_more_concepts_accuracy["correct"] += 1

    concept_count_accuracy = {
        count: round(acc["correct"] / acc["total"], 4)
        for count, acc in concept_count_accuracy.items() if acc["total"] > 0
    }

    if two_or_more_concepts_accuracy["total"] > 0:
        two_or_more_concepts_accuracy = round(
            two_or_more_concepts_accuracy["correct"] / two_or_more_concepts_accuracy["total"], 4
        )
    else:
        two_or_more_concepts_accuracy = None

    return concept_count_accuracy, two_or_more_concepts_accuracy


def evaluate_accuracy_by_question(results):
    """ABQ: 所有子问题都对才算对"""
    correct = sum('correctness' in result and all(result['correctness'].values()) for result in results)
    total = len(results)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_by_sub_question(results):
    """UASQ: 每个子问题独立计算"""
    correct = sum(sum(result['correctness'].values()) for result in results if 'correctness' in result)
    total = sum(len(result['correctness']) for result in results if 'correctness' in result)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_proportional_by_sub_question(results):
    """PSAQ: 按子问题比例加权"""
    total_score = 0
    for result in results:
        if 'correctness' in result:
            sub_question_count = len(result['correctness'])
            score_per_sub_question = 1 / sub_question_count if sub_question_count > 0 else 0
            question_score = sum(result['correctness'].values()) * score_per_sub_question
            total_score += question_score
    return round(total_score / len(results), 4) if results else 0


def main():
    parser = argparse.ArgumentParser(description="计算 DABench 评估指标")
    parser.add_argument('--questions_file_path', type=str, required=True,
                        help='问题文件路径 (JSONL)')
    parser.add_argument('--labels_file_path', type=str, required=True,
                        help='标签文件路径 (JSONL)')
    parser.add_argument('--responses_file_path', type=str, required=True,
                        help='响应文件路径 (JSONL)')
    args = parser.parse_args()

    # 转换为绝对路径
    script_dir = Path(__file__).parent
    da_bench_dir = script_dir.parent

    questions_path = (script_dir / args.questions_file_path).resolve()
    labels_path = (script_dir / args.labels_file_path).resolve()

    # responses_path 相对于 da-bench 目录
    if Path(args.responses_file_path).is_absolute():
        responses_path = Path(args.responses_file_path)
    else:
        responses_path = (da_bench_dir / args.responses_file_path).resolve()

    # 读取文件
    questions = load_jsonl(str(questions_path))
    labels = load_jsonl(str(labels_path))
    responses = load_jsonl(str(responses_path))

    # 读取 concept 信息
    concepts_data = read_concepts_from_questions(questions)

    # 评估响应
    results = evaluate_responses(labels, responses)

    # 计算主要指标
    abq = evaluate_accuracy_by_question(results)
    uasq = evaluate_accuracy_by_sub_question(results)
    psaq = evaluate_accuracy_proportional_by_sub_question(results)

    # 计算 concept 分析
    concept_accuracy = analyze_concepts_accuracy(results, concepts_data)
    concept_count_accuracy, two_or_more_concepts = analyze_concepts_count_accuracy(results, concepts_data)

    # 打印主要结果
    print("=" * 80)
    print("DABench 评估结果")
    print("=" * 80)
    print(f"ABQ  (Accuracy by Question):                    {abq:.2%}")
    print(f"PSAQ (Proportional by Sub-Question):            {psaq:.2%}")
    print(f"UASQ (Accuracy by Sub-Question):                {uasq:.2%}")
    print("=" * 80)

    # 打印 concept 分析
    if concept_accuracy:
        print("\nConcept Accuracy Analysis:")
        for concept, accuracy in sorted(concept_accuracy.items()):
            print(f"  {concept}: {accuracy:.2%}")

    if concept_count_accuracy:
        print("\nConcept Count Accuracy Analysis:")
        for count, accuracy in sorted(concept_count_accuracy.items()):
            print(f"  {count} Concept(s): {accuracy:.2%}")

    if two_or_more_concepts is not None:
        print(f"\nAccuracy for Questions with >= 2 Concepts: {two_or_more_concepts:.2%}")

    # 保存详细结果
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_name = Path(args.responses_file_path).stem
    output_file = output_dir / f"{responses_name}_metrics.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "main_results": {
                "ABQ": abq,
                "PSAQ": psaq,
                "UASQ": uasq
            },
            "concept_accuracy": concept_accuracy,
            "concept_count_accuracy": concept_count_accuracy,
            "two_or_more_concepts_accuracy": two_or_more_concepts,
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存至: {output_file}")


if __name__ == "__main__":
    main()
