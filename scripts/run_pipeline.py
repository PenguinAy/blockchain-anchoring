"""
一键流程编排器。

按五阶段执行完整锚定流程：
    A — 环境自检
    B — 模型训练
    C — 哈希计算（Merkle 树 + 训练时序链）
    D — 合约部署 + 链上锚定
    E — 验证与溯源查询

用法：
    python run_pipeline.py              # 全流程
    python run_pipeline.py --from D     # 从阶段 D 恢复
    python run_pipeline.py --dry-run    # 预览，不执行
    python run_pipeline.py --force      # 强制重跑
"""

import subprocess
import sys
import os
import json
import time
import argparse
from pathlib import Path


# ========== 配置 ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # f:\python_work\blockchain
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
HARDHAT_DIR = PROJECT_ROOT / "blockchain_anchor"

RPC_URL = "http://127.0.0.1:8545"

# 各阶段产物（用于检测是否已完成）
ARTIFACTS = {
    "A": None,  # 环境自检，无产物
    "B": ["checkpoints/lenet_epoch1.pth", "checkpoints/lenet_epoch5.pth",
          "checkpoints/training_history.json"],
    "C": ["merkle_result.json", "weight_chain.json"],
    "D": ["deploy_info.json", "upload_report.json"],
    "E": None,  # 始终可跑
}

PHASE_ORDER = ["A", "B", "C", "D", "E"]
PHASE_LABELS = {
    "A": "环境自检（依赖项 + Hardhat node）",
    "B": "模型训练（LeNet-5 + MNIST）",
    "C": "哈希计算（Merkle Tree + 时序链）",
    "D": "合约部署 + 数据上链锚定",
    "E": "链上验证与溯源查询",
}

# ========== 工具函数 ==========
def banner(phase: str):
    """打印阶段横幅"""
    label = PHASE_LABELS[phase]
    print(f"\n{'='*60}")
    print(f"  阶段 {phase} — {label}")
    print(f"{'='*60}")


def step(msg: str, status: str = ""):
    tail = f"  [{status}]" if status else ""
    print(f"  {msg}{tail}")


def ok(msg: str = "完成"):   step(msg, "OK")
def warn(msg: str):           step(msg, "WARN")
def fail(msg: str):           step(msg, "FAIL")


def check_artifact(phase: str) -> bool:
    paths = ARTIFACTS.get(phase)
    if paths is None:
        return False
    return all((PROJECT_ROOT / p).exists() for p in paths)


def run_python(script_name: str, extra_args: list[str] | None = None) -> int:
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def run_hardhat(args: list[str]) -> int:
    # Windows 下 npx 是 npx.cmd，需 shell=True 才能找到
    is_win = sys.platform == "win32"
    cmd = "npx hardhat " + " ".join(args)
    return subprocess.run(cmd, cwd=str(HARDHAT_DIR), shell=is_win).returncode


def is_hardhat_node_running() -> bool:
    try:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        return w3.is_connected()
    except Exception as e:
        # 调试：打印实际错误
        import traceback
        print(f"  [DEBUG] Hardhat 连接异常: {e}")
        return False


# ========== 阶段 A：环境自检 ==========
def phase_a(args) -> bool:
    banner("A")

    imports_ok = True
    for mod in ["torch", "torchvision", "web3", "Crypto", "numpy", "matplotlib"]:
        try:
            __import__(mod)
            step(f"import {mod:20s}", "OK")
        except ImportError:
            step(f"import {mod:20s}", "MISSING")
            imports_ok = False

    for d in ["blockchain_anchor/contracts", "checkpoints", "scripts"]:
        path = PROJECT_ROOT / d
        step(f"目录 {d:30s}", "OK" if path.is_dir() else "MISSING")

    is_win = sys.platform == "win32"
    proc = subprocess.run("npx hardhat --version" if is_win else ["npx", "hardhat", "--version"],
                          cwd=str(HARDHAT_DIR), capture_output=True, text=True,
                          shell=is_win)
    hh_ok = proc.returncode == 0
    step(f"Hardhat {'可用' if hh_ok else '不可用':30s}", "OK" if hh_ok else "FAIL")

    if is_hardhat_node_running():
        step("Hardhat 本地链 (8545)", "OK")
    else:
        step("Hardhat 本地链 (8545)", "OFFLINE")

    if not imports_ok:
        fail("缺少 Python 依赖，请检查 conda 环境")
        return False
    ok("环境自检通过")
    return True


# ========== 阶段 B：模型训练 ==========
def phase_b(args) -> bool:
    banner("B")
    if check_artifact("B") and not args.force:
        step("checkpoints 已存在，跳过（--force 强制重跑）")
        return True
    rc = run_python("train_lenet.py")
    if rc != 0:
        fail("训练失败，检查 train_lenet.py 输出")
        return False
    ok("checkpoints/lenet_epoch1~5.pth + training_curve.png")
    return True


# ========== 阶段 C：哈希计算 ==========
def phase_c(args) -> bool:
    banner("C")
    if check_artifact("C") and not args.force:
        step("merkle_result.json + weight_chain.json 已存在，跳过")
        return True

    step("构建 Merkle Tree...")
    if run_python("build_merkle.py") != 0:
        fail("build_merkle.py 失败")
        return False
    ok("merkle_result.json")

    step("构建训练时序哈希链...")
    if run_python("build_weight_chain.py") != 0:
        fail("build_weight_chain.py 失败")
        return False
    ok("weight_chain.json")
    return True


# ========== 阶段 D：合约部署 + 上链 ==========
def phase_d(args) -> bool:
    banner("D")

    # D.1 本地链（重试 3 次，每次间隔 1 秒）
    import time as _time
    node_ok = False
    for attempt in range(3):
        if is_hardhat_node_running():
            node_ok = True
            break
        if attempt < 2:
            step(f"等待本地链就绪 (尝试 {attempt+1}/3)...")
            _time.sleep(1)
    if not node_ok:
        warn("Hardhat 本地链未启动！")
        print(f"\n  请在另一个终端执行：")
        print(f"    cd {HARDHAT_DIR}")
        print(f"    npx hardhat node")
        print(f"\n  启动后按 Enter 继续...")
        input()
        if not is_hardhat_node_running():
            fail("仍未检测到本地链")
            return False
    ok("本地链运行中")

    # D.2 编译
    step("编译 Solidity 合约...")
    if run_hardhat(["compile"]) != 0:
        fail("合约编译失败")
        return False
    ok("编译通过")

    # D.3 部署
    deploy_info = PROJECT_ROOT / "deploy_info.json"
    if deploy_info.exists() and not args.force:
        step("deploy_info.json 已存在，跳过部署")
    else:
        step("部署合约到本地链...")
        if run_hardhat(["run", "scripts/deploy.ts", "--network", "localhost"]) != 0:
            fail("合约部署失败")
            return False
        if not deploy_info.exists():
            fail("deploy_info.json 未生成")
            return False
    ok(f"合约已部署 → {deploy_info}")

    # D.4 上链
    if check_artifact("D") and not args.force:
        step("upload_report.json 已存在，跳过上链")
        return True

    step("锚定数据+权重到链上...")
    if run_python("upload_to_chain.py") != 0:
        fail("上链失败")
        return False
    ok("upload_report.json — 数据+权重已锚定")
    return True


# ========== 阶段 E：验证查询 ==========
def phase_e(args) -> bool:
    banner("E")
    if not is_hardhat_node_running():
        fail("Hardhat 本地链未运行")
        return False

    step("单条样本 Merkle Proof 验证...")
    rc1 = run_python("verify_single.py")
    if rc1 != 0:
        warn("verify_single.py 有错误，继续...")

    step("正向溯源（数据 → 模型）...")
    rc2 = run_python("query_models_by_dataset.py")
    if rc2 != 0:
        warn("query_models_by_dataset.py 有错误")

    step("反向溯源（模型 → 数据 + 血缘）...")
    rc3 = run_python("query_dataset_by_model.py")
    if rc3 != 0:
        warn("query_dataset_by_model.py 有错误")

    ok("阶段 E 完成")
    return True


# ========== 阶段函数注册 ==========
PHASE_FUNCTIONS = {
    "A": phase_a,
    "B": phase_b,
    "C": phase_c,
    "D": phase_d,
    "E": phase_e,
}


# ========== 主流程 ==========
def main():
    parser = argparse.ArgumentParser(
        description="流程管控（阶段 A→B→C→D→E）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run_pipeline.py              全流程（A→B→C→D→E）
  python run_pipeline.py --from B     跳过环境检查，从训练开始
  python run_pipeline.py --from D     仅上链+验证
  python run_pipeline.py --force      强制重跑所有阶段
  python run_pipeline.py --dry-run    预览产物状态，不执行
        """
    )
    parser.add_argument("--from", dest="start_from", default="A",
                        choices=PHASE_ORDER, metavar="PHASE",
                        help=f"起始阶段 (默认 A)")
    parser.add_argument("--to", dest="end_at", default="E",
                        choices=PHASE_ORDER, metavar="PHASE",
                        help=f"结束阶段 (默认 E)")
    parser.add_argument("--force", action="store_true",
                        help="强制重跑所有阶段")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检查产物状态，不执行")
    args = parser.parse_args()

    print("=" * 60)
    print("  区块链锚定 · 一键流程管控")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  运行时:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"  阶段: A(环境) → B(训练) → C(哈希) → D(上链) → E(验证)")
    print("=" * 60)

    # --- Dry-run ---
    if args.dry_run:
        print("\n[Dry-Run] 仅检查产物状态\n")
        for phase in PHASE_ORDER:
            label = PHASE_LABELS[phase]
            paths = ARTIFACTS.get(phase)
            if paths is None:
                print(f"  [阶段 {phase}] {label}: 无产物检测")
            else:
                found = all((PROJECT_ROOT / p).exists() for p in paths)
                status = "DONE" if found else "PENDING"
                missing = [p for p in paths if not (PROJECT_ROOT / p).exists()]
                print(f"  [阶段 {phase}] {label}: {status}")
                if missing:
                    print(f"         缺失: {', '.join(missing)}")
        print(f"\n  Hardhat node: {'ONLINE' if is_hardhat_node_running() else 'OFFLINE'}")
        return

    # --- 确定执行范围 ---
    start_idx = PHASE_ORDER.index(args.start_from)
    end_idx = PHASE_ORDER.index(args.end_at)
    if start_idx > end_idx:
        print(f"[!] --from {args.start_from} 在 --to {args.end_at} 之后")
        sys.exit(1)

    phases_to_run = PHASE_ORDER[start_idx:end_idx + 1]
    print(f"\n执行范围: {' → '.join(phases_to_run)}")
    if args.force:
        print("模式:     强制重跑 (--force)")

    # --- 逐阶段执行 ---
    results = {}
    for phase in phases_to_run:
        fn = PHASE_FUNCTIONS[phase]
        try:
            passed = fn(args)
        except KeyboardInterrupt:
            print(f"\n\n[!] 用户在阶段 {phase} 中断")
            sys.exit(1)
        except Exception as e:
            fail(f"阶段 {phase} 异常: {e}")
            passed = False

        results[phase] = passed
        if not passed:
            fail(f"阶段 {phase} 失败，流水线中断")
            print(f"  修复后可从断点继续: python run_pipeline.py --from {phase}")
            break

    # --- 汇总 ---
    print(f"\n{'='*60}")
    print("  流水线执行汇总")
    print(f"{'='*60}")
    for phase in phases_to_run:
        if phase in results:
            status = "通过" if results[phase] else "失败"
        else:
            status = "— 未执行"
        print(f"  [阶段 {phase}] {PHASE_LABELS[phase]:30s} {status}")

    all_pass = all(results.get(p, False) for p in phases_to_run)
    if all_pass and "E" in results:
        print(f"\n  全流程完成。")
        report_path = PROJECT_ROOT / "upload_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            print(f"  数据注册 Gas:  {report.get('data_gas_used', 'N/A')}")
            print(f"  权重注册 Gas:  {report.get('weight_gas_used', 'N/A')}")
            print(f"  链上验证:      {report.get('data_verification', 'N/A')}")
            print(f"  篡改检测:      {report.get('tamper_detection', 'N/A')}")
    else:
        failed_phases = [p for p, ok in results.items() if not ok]
        if failed_phases:
            print(f"\n  失败阶段: {', '.join(failed_phases)}")
            print(f"  从断点恢复: python run_pipeline.py --from {failed_phases[0]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
