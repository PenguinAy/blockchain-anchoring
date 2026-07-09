"""
Contract deployment utilities.

Deploys DataAnchor and WeightAnchor contracts to any EVM-compatible
chain without requiring Hardhat or Node.js at runtime.  Requires
pre-compiled bytecode (from ``npx hardhat compile`` or solc directly).
"""

import json
from pathlib import Path
from typing import Optional
from web3 import Web3


def deploy_contract(
    w3: Web3,
    private_key: str,
    contract_name: str,
    artifacts_dir: str | Path,
    constructor_args: list | None = None,
    gas: int = 2_000_000,
) -> str:
    """
    Deploy a single Solidity contract from its Hardhat build artifact.

    Parameters
    ----------
    w3 : Web3
        Connected Web3 instance.
    private_key : str
        Hex-encoded private key of the deployer account.
    contract_name : str
        Contract name as it appears in the artifacts directory
        (e.g. ``"DataAnchor"``).
    artifacts_dir : str or Path
        Path to the Hardhat artifacts directory containing the compiled
        ``.json`` files.
    constructor_args : list, optional
        Arguments passed to the contract constructor.
    gas : int
        Gas limit for the deployment transaction.

    Returns
    -------
    str
        Checksum address of the deployed contract.

    Raises
    ------
    FileNotFoundError
        If the artifact file does not exist.
    """
    artifacts = Path(artifacts_dir)
    artifact_path = (
        artifacts / "contracts" / f"{contract_name}.sol" / f"{contract_name}.json"
    )
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    with open(artifact_path) as f:
        artifact = json.load(f)

    abi = artifact["abi"]
    bytecode = artifact["bytecode"]

    account = w3.eth.account.from_key(private_key)
    ContractFactory = w3.eth.contract(abi=abi, bytecode=bytecode)

    tx = ContractFactory.constructor(
        *(constructor_args or [])
    ).build_transaction({
        "from": account.address,
        "nonce": w3.eth.get_transaction_count(account.address),
        "gas": gas,
        "gasPrice": w3.eth.gas_price,
        "chainId": w3.eth.chain_id,
    })

    signed = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.contractAddress


def deploy_anchors(
    rpc_url: str,
    private_key: str,
    artifacts_dir: str | Path = "blockchain_anchor/artifacts",
) -> tuple[str, str]:
    """
    Deploy both DataAnchor and WeightAnchor contracts.

    Convenience wrapper that deploys the two core contracts in sequence.

    Parameters
    ----------
    rpc_url : str
        JSON-RPC endpoint.
    private_key : str
        Deployer account private key (hex).
    artifacts_dir : str or Path
        Path to compiled Hardhat artifacts.

    Returns
    -------
    tuple[str, str]
        (data_anchor_address, weight_anchor_address) as checksum strings.
    """
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to {rpc_url}")

    account = w3.eth.account.from_key(private_key)
    balance = w3.from_wei(w3.eth.get_balance(account.address), "ether")
    print(f"Deployer:  {account.address}")
    print(f"Balance:   {balance} ETH")
    print(f"Chain ID:  {w3.eth.chain_id}")

    print("\n[1/2] Deploying DataAnchor...")
    data_addr = deploy_contract(w3, private_key, "DataAnchor", artifacts_dir)
    print(f"  Address: {data_addr}")

    print("[2/2] Deploying WeightAnchor...")
    weight_addr = deploy_contract(w3, private_key, "WeightAnchor", artifacts_dir)
    print(f"  Address: {weight_addr}")

    return data_addr, weight_addr
