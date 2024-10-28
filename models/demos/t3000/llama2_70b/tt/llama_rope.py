import torch
import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor
from models.common.lightweightmodule import LightweightModule
from models.demos.t3000.llama2_70b.tt.llama_common import precompute_freqs


def get_rot_transformation_mat(dhead):
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def compute_gather_cos_sin(dhead, end, theta, position_ids):
    cos, sin = precompute_freqs(dhead, end, theta)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


class TtLlamaRotarySetup(LightweightModule):
    def __init__(
        self,
        device,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.device = device

        self.core_grid = device.compute_with_storage_grid_size()
        num_cores = self.core_grid.x * self.core_grid.y

        # Generate the cos/sin matrices needed for ttnn.embedding op
        cos_matrix, sin_matrix = compute_gather_cos_sin(
            dhead=head_dim, end=max_seq_len * 2, theta=rope_theta, position_ids=torch.arange(max_seq_len)
        )

        self.cos_matrix = ttnn.from_torch(
            cos_matrix,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device),
        )
        self.sin_matrix = ttnn.from_torch(
            sin_matrix,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            mesh_mapper=ReplicateTensorToMesh(device),
        )

        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1, 1, num_cores, 1
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, ttnn.TILE_SIZE * num_cores, ttnn.TILE_SIZE),
            core_grid=ttnn.CoreGrid(y=self.core_grid.y, x=self.core_grid.x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat, device=device, layout=ttnn.TILE_LAYOUT, dtype=datatype, memory_config=trans_mat_mem_config
        )

    def get_trans_mats(self):
        return self.transformation_mat

    def get_rot_mats(self, position_idxs):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"

        batch = position_idxs.shape[0]
        position_idxs = position_idxs.unsqueeze(-1).unsqueeze(-1)
        assert position_idxs.shape == (batch, 1, 1), "position idxs must be a [batch, 1] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        position_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ReplicateTensorToMesh(self.device),
        )

        cos = ttnn.embedding(position_idxs, self.cos_matrix)  # [batch, head_dim, head_dim]
        sin = ttnn.embedding(position_idxs, self.sin_matrix)  # [batch, head_dim, head_dim]

        cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
        sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT)

        grid = (
            ttnn.CoreRangeSet(ttnn.num_cores_to_corerange_set(batch, self.core_grid, row_wise=True))
            .bounding_box()
            .grid_size()
        )
        mem_config = ttnn.create_sharded_memory_config(
            shape=(1, batch, ttnn.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

        return [cos, sin]
