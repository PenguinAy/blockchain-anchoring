import { createWalletClient, createPublicClient, http, formatEther } from "viem";
import { hardhat } from "viem/chains";
import { privateKeyToAccount } from "viem/accounts";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

// ESM 兼容：用 import.meta.url 替代 __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Hardhat 本地链 Account #0 私钥
const PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
const contractsDir = path.resolve(__dirname, "..", "artifacts", "contracts");

function loadArtifact(name: string) {
  const p = path.join(contractsDir, `${name}.sol`, `${name}.json`);
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

async function main() {
  console.log("========================================");
  console.log("  区块链锚定 — 合约部署");
  console.log("========================================\n");

  const account = privateKeyToAccount(PRIVATE_KEY as `0x${string}`);
  const walletClient = createWalletClient({
    account,
    chain: hardhat,
    transport: http("http://127.0.0.1:8545"),
  });
  const publicClient = createPublicClient({
    chain: hardhat,
    transport: http("http://127.0.0.1:8545"),
  });

  const balance = await publicClient.getBalance({ address: account.address });
  console.log(`[部署账户] ${account.address}`);
  console.log(`[账户余额] ${formatEther(balance)} ETH\n`);

  // --- 部署 DataAnchor ---
  console.log("[1/2] 部署 DataAnchor...");
  const dataArtifact = loadArtifact("DataAnchor");
  const dataHash = await walletClient.deployContract({
    abi: dataArtifact.abi,
    bytecode: dataArtifact.bytecode as `0x${string}`,
    account: account,
  });
  const dataReceipt = await publicClient.waitForTransactionReceipt({ hash: dataHash });
  console.log(`  ✓ DataAnchor 已部署`);
  console.log(`  地址: ${dataReceipt.contractAddress}`);
  console.log(`  TX Hash: ${dataHash}`);

  // --- 部署 WeightAnchor ---
  console.log("\n[2/2] 部署 WeightAnchor...");
  const weightArtifact = loadArtifact("WeightAnchor");
  const weightHash = await walletClient.deployContract({
    abi: weightArtifact.abi,
    bytecode: weightArtifact.bytecode as `0x${string}`,
    account: account,
  });
  const weightReceipt = await publicClient.waitForTransactionReceipt({ hash: weightHash });
  console.log(`  ✓ WeightAnchor 已部署`);
  console.log(`  地址: ${weightReceipt.contractAddress}`);
  console.log(`  TX Hash: ${weightHash}`);

  // --- 汇总 ---
  const dataAddr = dataReceipt.contractAddress!;
  const weightAddr = weightReceipt.contractAddress!;

  console.log("\n========================================");
  console.log("  部署完成！");
  console.log("========================================");
  console.log(`  DataAnchor:     ${dataAddr}`);
  console.log(`  WeightAnchor:   ${weightAddr}`);
  console.log(`  部署账户:       ${account.address}`);
  console.log(`  网络:           localhost (31337)`);
  console.log("========================================\n");

  const deployInfo = {
    dataAnchor: dataAddr,
    weightAnchor: weightAddr,
    deployer: account.address,
    network: "localhost",
    chainId: 31337,
  };
  console.log("[DEPLOY_INFO_JSON]");
  console.log(JSON.stringify(deployInfo, null, 2));

  const outPath = path.resolve(__dirname, "..", "..", "deploy_info.json");
  fs.writeFileSync(outPath, JSON.stringify(deployInfo, null, 2));
  console.log(`\n[✓] 部署信息已写入: ${outPath}`);
}

main().catch((error) => {
  console.error("部署失败:", error);
  process.exit(1);
});
