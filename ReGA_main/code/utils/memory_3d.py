import torch
import numpy as np
from numpy.linalg import norm


class Memory(object):
    """
        Create the empty memory buffer for 3D features
    """

    def __init__(self, size, dimension=1 * 4 * 128 * 128 * 128):
        self.memory = {}
        self.size = size
        self.dimension = dimension
        print(f"Memory initialized: size={size}, dimension={dimension}")

    def reset(self):
        self.memory = {}
        print("Memory bank reset")

    def get_size(self):
        return len(self.memory)

    def push(self, keys, logits):
        """
        Push keys and corresponding logits into memory
        """
        try:
            for i, key in enumerate(keys):
                # 检查 memory 大小
                if len(self.memory.keys()) >= self.size:
                    # 移除最老的项
                    oldest_key = list(self.memory.keys())[0]
                    self.memory.pop(oldest_key)

                # 展平 key
                try:
                    key_flat = key.reshape(self.dimension)
                except ValueError:
                    # 如果维度不匹配，调整维度
                    actual_dim = np.prod(key.shape)
                    print(f"Warning: Key dimension mismatch. Expected {self.dimension}, got {actual_dim}")
                    key_flat = key.reshape(actual_dim)

                # 确保有对应的 logit
                if i < len(logits):
                    logit = logits[i]
                else:
                    # 如果没有足够的 logits，使用第一个或创建默认值
                    logit = logits[0] if len(logits) > 0 else np.zeros(1, dtype=np.float32)

                # 存储到 memory
                self.memory.update({key_flat.tobytes(): logit})

        except Exception as e:
            print(f"Error in memory push: {e}")
            print(f"Keys shape: {keys.shape if hasattr(keys, 'shape') else 'N/A'}")
            print(f"Logits shape: {logits.shape if hasattr(logits, 'shape') else 'N/A'}")

    def _prepare_batch(self, sample, attention_weight):
        """
        Prepare batch by weighted averaging of samples
        """
        try:
            # 确保 attention_weight 是 numpy 数组
            attention_weight = np.array(attention_weight, dtype=np.float32)

            # 防止除零
            if np.sum(attention_weight) == 0:
                attention_weight = np.ones_like(attention_weight) / len(attention_weight)
            else:
                # 应用 softmax
                attention_weight = np.exp(attention_weight / 0.2)
                attention_weight = attention_weight / (np.sum(attention_weight) + 1e-8)

            # 检查 sample 是否为空
            if len(sample) == 0:
                print("Warning: sample is empty in _prepare_batch")
                return torch.zeros(1, dtype=torch.float32)

            # 确保所有 sample 是 numpy 数组
            sample_arrays = []
            for s in sample:
                if isinstance(s, np.ndarray):
                    sample_arrays.append(s.astype(np.float32))
                elif isinstance(s, torch.Tensor):
                    sample_arrays.append(s.cpu().numpy().astype(np.float32))
                else:
                    # 尝试转换
                    try:
                        sample_arrays.append(np.array(s, dtype=np.float32))
                    except:
                        print(f"Warning: Could not convert sample type {type(s)} to numpy")
                        sample_arrays.append(np.zeros(1, dtype=np.float32))

            # 加权平均
            if len(sample_arrays) == 1:
                ensemble_prediction = sample_arrays[0]
            else:
                ensemble_prediction = sample_arrays[0] * attention_weight[0]
                for i in range(1, len(sample_arrays)):
                    ensemble_prediction = ensemble_prediction + sample_arrays[i] * attention_weight[i]

            return torch.FloatTensor(ensemble_prediction)

        except Exception as e:
            print(f"Error in _prepare_batch: {e}")
            return torch.zeros(1, dtype=torch.float32)

    def get_neighbours(self, keys, k):
        """
        Returns samples from buffer using nearest neighbour approach
        """
        try:
            samples = []
            similarity_scores_list = []

            # 展平 keys
            keys = keys.reshape(len(keys), self.dimension)
            total_keys = len(self.memory.keys())

            # 检查 memory 是否为空
            if total_keys == 0:
                print("Memory bank is empty, returning default values")
                # 返回默认值
                for _ in keys:
                    default_value = np.zeros(self.dimension, dtype=np.float32)
                    samples.append(torch.FloatTensor(default_value))
                return torch.stack(samples), 0.0

            # 从 memory 中获取所有 keys
            self.all_keys = np.frombuffer(
                np.asarray(list(self.memory.keys())), dtype=np.float32
            ).reshape(total_keys, self.dimension)

            for key in keys:
                # 计算相似性得分
                similarity_scores = np.dot(self.all_keys, key.T) / (
                        norm(self.all_keys, axis=1) * norm(key.T) + 1e-8
                )

                # 记录平均相似度
                similarity_scores_list.append(np.mean(similarity_scores))

                # 确保 k 不超过可用键的数量
                k_actual = min(k, len(similarity_scores))

                if k_actual > 0:
                    # 获取 k 个最近邻居
                    indices = np.argpartition(similarity_scores, -k_actual)[-k_actual:]
                    K_neighbour_keys = self.all_keys[indices]

                    # 获取邻居的值
                    neighbours = []
                    for nkey in K_neighbour_keys:
                        nkey_bytes = nkey.tobytes()
                        if nkey_bytes in self.memory:
                            neighbours.append(self.memory[nkey_bytes])
                        else:
                            print(f"Warning: Key not found in memory")
                            neighbours.append(np.zeros(1, dtype=np.float32))

                    # 计算注意力权重
                    attention_weight = similarity_scores[indices]

                    # 准备批次
                    batch = self._prepare_batch(neighbours, attention_weight)
                    samples.append(batch)
                else:
                    # 如果没有邻居，返回零张量
                    samples.append(torch.FloatTensor(np.zeros(self.dimension, dtype=np.float32)))

            # 返回结果和平均相似度
            avg_similarity = np.mean(similarity_scores_list) if similarity_scores_list else 0.0
            return torch.stack(samples), avg_similarity

        except Exception as e:
            print(f"Error in get_neighbours: {e}")
            import traceback
            traceback.print_exc()

            # 返回默认值
            default_samples = []
            for _ in range(len(keys)):
                default_samples.append(torch.zeros(self.dimension, dtype=torch.float32))
            return torch.stack(default_samples), 0.0