// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title 训练数据集链上锚定合约
/// @notice 通过 Merkle Tree 根哈希存证训练数据完整性，
///         支持单条成员证明、子集多证明、Sparse Merkle Tree 差集证明
contract DataAnchor {
    struct DatasetRecord {
        bytes32 merkleRoot;     // 数据集 Merkle Tree 根哈希
        bytes32 smtRoot;        // Sparse Merkle Tree 根（用于非成员证明，可选）
        string  metadata;       // 元数据（数据集名/样本数/IPFS CID）
        address registrar;      // 注册者
        uint256 timestamp;      // 上链时间戳
    }

    mapping(bytes32 => DatasetRecord) public datasets;
    bytes32[] public datasetIds;

    event DatasetRegistered(
        bytes32 indexed datasetId,
        bytes32 merkleRoot,
        address indexed registrar
    );

    event DatasetBatchRegistered(
        uint256 count,
        address indexed registrar
    );

    // ========== 基础注册 ==========

    /// @notice 注册一个数据集
    function registerDataset(
        bytes32 _datasetId,
        bytes32 _merkleRoot,
        string calldata _metadata
    ) external {
        require(datasets[_datasetId].timestamp == 0, "Dataset already exists");
        datasets[_datasetId] = DatasetRecord({
            merkleRoot: _merkleRoot,
            smtRoot: bytes32(0),
            metadata: _metadata,
            registrar: msg.sender,
            timestamp: block.timestamp
        });
        datasetIds.push(_datasetId);
        emit DatasetRegistered(_datasetId, _merkleRoot, msg.sender);
    }

    /// @notice 注册数据集（含 Sparse Merkle Tree 根，用于非成员证明）
    function registerDatasetWithSMT(
        bytes32 _datasetId,
        bytes32 _merkleRoot,
        bytes32 _smtRoot,
        string calldata _metadata
    ) external {
        require(datasets[_datasetId].timestamp == 0, "Dataset already exists");
        datasets[_datasetId] = DatasetRecord({
            merkleRoot: _merkleRoot,
            smtRoot: _smtRoot,
            metadata: _metadata,
            registrar: msg.sender,
            timestamp: block.timestamp
        });
        datasetIds.push(_datasetId);
        emit DatasetRegistered(_datasetId, _merkleRoot, msg.sender);
    }

    /// @notice 批量注册数据集（Gas 优化：分摊 21000 固定开销）
    function batchRegisterDataset(
        bytes32[] calldata _ids,
        bytes32[] calldata _roots,
        string[] calldata _metadatas
    ) external {
        uint256 n = _ids.length;
        require(n == _roots.length && n == _metadatas.length, "Length mismatch");
        for (uint256 i = 0; i < n; i++) {
            if (datasets[_ids[i]].timestamp == 0) {
                datasets[_ids[i]] = DatasetRecord({
                    merkleRoot: _roots[i],
                    smtRoot: bytes32(0),
                    metadata: _metadatas[i],
                    registrar: msg.sender,
                    timestamp: block.timestamp
                });
                datasetIds.push(_ids[i]);
            }
        }
        emit DatasetBatchRegistered(n, msg.sender);
    }

    // ========== 数据集完整性验证 ==========

    /// @notice 验证数据集完整性（本地根 vs 链上根）
    function verifyDataset(
        bytes32 _datasetId,
        bytes32 _localRoot
    ) external view returns (bool) {
        return datasets[_datasetId].merkleRoot == _localRoot;
    }

    // ========== 单条样本成员证明（经典 Merkle Proof）==========

    /// @notice 验证单条数据是否属于数据集
    /// @dev 兄弟节点按字典序排序后拼接 keccak256，与 OpenZeppelin 一致
    function verifyMember(
        bytes32 _datasetId,
        bytes32 _leaf,
        bytes32[] calldata _proof
    ) external view returns (bool) {
        bytes32 computedHash = _leaf;
        for (uint256 i = 0; i < _proof.length; i++) {
            bytes32 sibling = _proof[i];
            computedHash = computedHash < sibling
                ? keccak256(abi.encodePacked(computedHash, sibling))
                : keccak256(abi.encodePacked(sibling, computedHash));
        }
        return computedHash == datasets[_datasetId].merkleRoot;
    }

    /// @notice 链下自行计算 Merkle Proof 后的纯验证（不依赖合约存储）
    /// @dev 可用于 Gas 为 0 的本地验证：拿到链上 root 后本地算 proof 走此函数
    function verifyMemberAgainstRoot(
        bytes32 _root,
        bytes32 _leaf,
        bytes32[] calldata _proof
    ) public pure returns (bool) {
        bytes32 computedHash = _leaf;
        for (uint256 i = 0; i < _proof.length; i++) {
            bytes32 sibling = _proof[i];
            computedHash = computedHash < sibling
                ? keccak256(abi.encodePacked(computedHash, sibling))
                : keccak256(abi.encodePacked(sibling, computedHash));
        }
        return computedHash == _root;
    }

    // ========== 子集多证明（Multi-Proof）==========

    /// @notice 验证多条数据是否同时属于数据集（批量 Merkle Proof）
    /// @dev 每条独立验证，任一失败返回 false
    function verifySubset(
        bytes32 _datasetId,
        bytes32[] calldata _leaves,
        bytes32[][] calldata _proofs
    ) external view returns (bool[] memory) {
        require(_leaves.length == _proofs.length, "Length mismatch");
        bytes32 root = datasets[_datasetId].merkleRoot;
        bool[] memory results = new bool[](_leaves.length);
        for (uint256 i = 0; i < _leaves.length; i++) {
            results[i] = verifyMemberAgainstRoot(root, _leaves[i], _proofs[i]);
        }
        return results;
    }

    /// @notice 子集全包含验证：所有叶子是否全部属于数据集
    function verifySubsetAll(
        bytes32 _datasetId,
        bytes32[] calldata _leaves,
        bytes32[][] calldata _proofs
    ) external view returns (bool) {
        bytes32 root = datasets[_datasetId].merkleRoot;
        for (uint256 i = 0; i < _leaves.length; i++) {
            if (!verifyMemberAgainstRoot(root, _leaves[i], _proofs[i])) {
                return false;
            }
        }
        return true;
    }

    // ========== 差集（非成员）证明 — Sparse Merkle Tree ==========

    /// @notice 验证某个 key 不在数据集中（非成员证明）
    /// @dev Sparse Merkle Tree 深度 256，空叶子默认为零值
    /// @param _smtRoot Sparse Merkle Tree 根哈希
    /// @param _key 待证明"不存在"的键（通常是数据哈希）
    /// @param _siblingPath 从叶到根的 256 个兄弟节点
    function verifyNonMembership(
        bytes32 _smtRoot,
        bytes32 _key,
        bytes32[256] calldata _siblingPath
    ) public pure returns (bool) {
        // 将 key 转为 256 位路径
        uint256 path = uint256(keccak256(abi.encodePacked(_key)));
        // 空叶子的默认值（SMT 中未占用叶子为零）
        bytes32 leaf = bytes32(0);

        bytes32 computedHash = leaf;
        for (uint256 i = 0; i < 256; i++) {
            // 从最低位开始（或最高位均可，只要与链下一致）
            bool bit = (path & (1 << (255 - i))) != 0;
            bytes32 sibling = _siblingPath[i];
            if (bit) {
                computedHash = keccak256(abi.encodePacked(sibling, computedHash));
            } else {
                computedHash = keccak256(abi.encodePacked(computedHash, sibling));
            }
        }
        return computedHash == _smtRoot;
    }

    // ========== 查询接口 ==========

    function getDatasetCount() external view returns (uint256) {
        return datasetIds.length;
    }

    function getDataset(bytes32 _datasetId) external view returns (DatasetRecord memory) {
        require(datasets[_datasetId].timestamp != 0, "Dataset not found");
        return datasets[_datasetId];
    }
}
