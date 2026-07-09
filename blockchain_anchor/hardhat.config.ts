import { defineConfig } from "hardhat/config";

export default defineConfig({
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    // Hardhat 内置本地链（默认），运行 npx hardhat node 启动
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
      type: "http",
    },
    // Ganache GUI 本地链（可选）
    ganache: {
      url: "http://127.0.0.1:7545",
      chainId: 1337,
      type: "http",
    },
  },
  paths: {
    sources: "./contracts",
    artifacts: "./artifacts",
    cache: "./cache",
  },
});
