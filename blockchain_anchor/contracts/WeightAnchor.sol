// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title 模型权重链上锚定合约
/// @notice 锚定模型权重哈希、训练过程链尾、版本链 DAG，
///         绑定训练数据来源，支持双向溯源与血缘查询
contract WeightAnchor {
    struct ModelRecord {
        bytes32 weightHash;      // 最终权重 SHA-256
        bytes32 datasetId;       // 训练数据集 ID（双向绑定）
        bytes32 chainTip;        // 训练时序哈希链尾 C_E
        bytes32 parentModelId;   // 父模型 ID（0 表示根模型，无父）
        string  modelMeta;       // 模型架构/超参/IPFS CID
        address owner;           // 注册者
        uint256 timestamp;       // 上链时间戳
        string  version;         // 语义版本号 "v1.0"
    }

    mapping(bytes32 => ModelRecord) public models;
    mapping(bytes32 => bytes32[]) public datasetToModels;  // 数据集 → 模型列表
    mapping(bytes32 => bytes32[]) public parentToChildren;  // 父模型 → 直接子模型列表
    bytes32[] public modelIds;

    event ModelRegistered(
        bytes32 indexed modelId,
        bytes32 weightHash,
        bytes32 indexed datasetId,
        bytes32 indexed parentModelId,
        bytes32 chainTip,
        address owner
    );

    // ========== 模型注册 ==========

    /// @notice 注册一个模型实例
    /// @param _modelId       全局唯一模型 ID = SHA256(weightHash || parentModelId || metadata)
    /// @param _weightHash    最终权重的 SHA-256 摘要
    /// @param _datasetId     关联的训练数据集 ID（需已在 DataAnchor 注册）
    /// @param _chainTip      Training hash chain tail C_E
    /// @param _parentModelId Parent model ID (0 = root model)
    /// @param _modelMeta     模型架构/超参摘要（可含 IPFS CID）
    /// @param _version       语义版本号
    function registerModel(
        bytes32 _modelId,
        bytes32 _weightHash,
        bytes32 _datasetId,
        bytes32 _chainTip,
        bytes32 _parentModelId,
        string calldata _modelMeta,
        string calldata _version
    ) external {
        require(models[_modelId].timestamp == 0, "Model already exists");

        // 如果指定了父模型，父模型必须已注册
        if (_parentModelId != bytes32(0)) {
            require(models[_parentModelId].timestamp != 0, "Parent model not found");
        }

        models[_modelId] = ModelRecord({
            weightHash:    _weightHash,
            datasetId:     _datasetId,
            chainTip:      _chainTip,
            parentModelId: _parentModelId,
            modelMeta:     _modelMeta,
            owner:         msg.sender,
            timestamp:     block.timestamp,
            version:       _version
        });

        datasetToModels[_datasetId].push(_modelId);

        // 版本链 DAG：注册为父模型的子节点
        if (_parentModelId != bytes32(0)) {
            parentToChildren[_parentModelId].push(_modelId);
        }

        modelIds.push(_modelId);

        emit ModelRegistered(
            _modelId, _weightHash, _datasetId,
            _parentModelId, _chainTip, msg.sender
        );
    }

    // ========== 权重完整性验证 ==========

    /// @notice 验证模型权重是否与链上记录一致
    function verifyModel(
        bytes32 _modelId,
        bytes32 _localWeightHash
    ) external view returns (bool) {
        require(models[_modelId].timestamp != 0, "Model not found");
        return models[_modelId].weightHash == _localWeightHash;
    }

    // ========== 训练过程审计 ==========

    /// @notice 验证训练时序哈希链尾（C_E）
    /// @dev 链下重新计算 C_t = H(W_t || C_{t-1}) 后与链上 chainTip 比对
    function verifyTrainingChain(
        bytes32 _modelId,
        bytes32 _localChainTip
    ) external view returns (bool) {
        require(models[_modelId].timestamp != 0, "Model not found");
        return models[_modelId].chainTip == _localChainTip;
    }

    /// @notice 验证单个 epoch checkpoint 是否在训练链中
    /// @dev 链下需提供从该 epoch 到链尾的哈希连（模拟 Merkle Proof）
    ///      给定 W_t, C_{t-1}, 以及后续的 (W_{t+1}, W_{t+2}, ..., W_E) 序列，
    ///      链下计算 C_E 后与此函数返回的链上 chainTip 比对
    function verifyEpochCheckpoint(
        bytes32 _modelId,
        bytes32 _epochHash,          // H(W_t)
        bytes32 _prevChainTip,       // C_{t-1}
        bytes32[] calldata _restWeights  // 后续 epoch 的 H(W_{t+1}) ... H(W_E)
    ) external view returns (bool) {
        require(models[_modelId].timestamp != 0, "Model not found");

        bytes32 chain = keccak256(abi.encodePacked(_epochHash, _prevChainTip));
        for (uint256 i = 0; i < _restWeights.length; i++) {
            chain = keccak256(abi.encodePacked(_restWeights[i], chain));
        }
        return chain == models[_modelId].chainTip;
    }

    // ========== 模型版本链 DAG 查询 ==========

    /// @notice 递归获取从模型到根模型的完整祖先链
    /// @dev 沿 parentModelId 链回溯，直到遇到 parentModelId == 0
    function getLineage(bytes32 _modelId)
        external view returns (bytes32[] memory)
    {
        require(models[_modelId].timestamp != 0, "Model not found");
        // 最多向上回溯 64 代（防无限循环，已超过实际需求）
        bytes32[] memory ancestors = new bytes32[](64);
        uint256 count = 0;
        bytes32 current = _modelId;
        while (count < 64) {
            bytes32 parent = models[current].parentModelId;
            if (parent == bytes32(0)) break;
            ancestors[count] = parent;
            count++;
            current = parent;
        }
        // 裁剪到实际长度
        bytes32[] memory result = new bytes32[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = ancestors[i];
        }
        return result;
    }

    /// @notice 获取直接子模型列表
    function getChildren(bytes32 _modelId)
        external view returns (bytes32[] memory)
    {
        return parentToChildren[_modelId];
    }

    /// @notice 递归获取所有子孙模型（BFS 展开 DAG）
    /// @dev maxDepth 限制搜索深度，返回所有子孙（非递归）
    function getDescendants(bytes32 _modelId, uint256 _maxDepth)
        external view returns (bytes32[] memory)
    {
        require(models[_modelId].timestamp != 0, "Model not found");
        // 最多返回 256 个子孙
        bytes32[] memory all = new bytes32[](256);
        bytes32[] memory queue = new bytes32[](256);
        uint256 resultCount = 0;
        uint256 head = 0;
        uint256 tail = 0;

        // BFS
        queue[tail++] = _modelId;
        for (uint256 depth = 0; depth < _maxDepth; depth++) {
            uint256 levelSize = tail - head;
            if (levelSize == 0) break;
            for (uint256 i = 0; i < levelSize; i++) {
                bytes32 current = queue[head++];
                if (current != _modelId) {
                    all[resultCount++] = current;
                }
                bytes32[] storage children = parentToChildren[current];
                for (uint256 j = 0; j < children.length; j++) {
                    if (tail < 256) {
                        queue[tail++] = children[j];
                    }
                }
            }
        }

        // 裁剪
        bytes32[] memory result = new bytes32[](resultCount);
        for (uint256 i = 0; i < resultCount; i++) {
            result[i] = all[i];
        }
        return result;
    }

    /// @notice 验证 childId 是否在 parentId 的祖先后代链上
    function verifyLineage(bytes32 _childId, bytes32 _ancestorId)
        external view returns (bool)
    {
        require(models[_childId].timestamp != 0, "Child model not found");
        require(models[_ancestorId].timestamp != 0, "Ancestor model not found");
        bytes32 current = _childId;
        for (uint256 i = 0; i < 64; i++) {
            bytes32 parent = models[current].parentModelId;
            if (parent == _ancestorId) return true;
            if (parent == bytes32(0)) return false;
            current = parent;
        }
        return false;
    }

    // ========== 双向溯源查询 ==========

    /// @notice 根据数据集 ID 查询所有用其训练的模型（正向溯源）
    function getModelsByDataset(bytes32 _datasetId)
        external view returns (bytes32[] memory)
    {
        return datasetToModels[_datasetId];
    }

    /// @notice 根据模型 ID 查询其训练数据来源（反向溯源）
    function getDatasetByModel(bytes32 _modelId)
        external view returns (bytes32)
    {
        require(models[_modelId].timestamp != 0, "Model not found");
        return models[_modelId].datasetId;
    }

    // ========== 查询接口 ==========

    function getModelCount() external view returns (uint256) {
        return modelIds.length;
    }

    function getModel(bytes32 _modelId) external view returns (ModelRecord memory) {
        require(models[_modelId].timestamp != 0, "Model not found");
        return models[_modelId];
    }
}
