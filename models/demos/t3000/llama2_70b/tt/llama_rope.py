import torch
import ttnn
from ttnn import ReplicateTensorToMesh
from models.common.lightweightmodule import LightweightModule
from models.demos.t3000.llama2_70b.tt.llama_common import precompute_freqs


def get_rot_transformation_mat(dhead):
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def compute_gather_cos_sin(dhead, end, position_ids):
    cos, sin = precompute_freqs(dhead, end)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin


class TtLlamaRotary(LightweightModule):
    def __init__(
        self,
        device,
        batch,
        head_dim: int,
        max_seq_len: int,
        mode: str,
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.batch = batch
        self.head_dim = head_dim
        self.device = device
        self.mode = mode

        self.core_grid = device.compute_with_storage_grid_size()
        num_cores = self.core_grid.x * self.core_grid.y

        if self.mode == "decode":
            # Generate the cos/sin matrices needed for ttnn.embedding op
            cos_matrix, sin_matrix = compute_gather_cos_sin(
                dhead=head_dim, end=max_seq_len * 2, position_ids=torch.arange(max_seq_len)
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

        else:
            self.transformation_mat = ttnn.from_torch(
                get_rot_transformation_mat(dhead=ttnn.TILE_SIZE), device=device, layout=ttnn.TILE_LAYOUT, dtype=datatype
            )

    def apply_rotary(self, x, cos, sin):
        # n_head = 8 for Q
        # n_head = 1 for K

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            # math_fidelity=ttnn.MathFidelity.LoFi,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=(True if self.head_dim <= 128 else False),
            packer_l1_acc=True,
        )

        rotary_output = ttnn.experimental.rotary_embedding_llama(
            x, cos, sin, self.transformation_mat, compute_kernel_config=compute_kernel_config
        )

        return rotary_output

    def prepare_cos_sin(self, position_ids):
        if self.mode == "decode":
            # assert isinstance(position_ids, torch.Tensor), "Position ids must be a torch tensor"

            # position_ids = position_ids.unsqueeze(-1)  # [batch, 1]
            # position_ids = ttnn.from_torch(
            #     position_ids, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=ReplicateTensorToMesh(self.device)
            # )

            cos = ttnn.embedding(position_ids, self.cos_matrix)  # [batch, head_dim, head_dim]
            sin = ttnn.embedding(position_ids, self.sin_matrix)  # [batch, head_dim, head_dim]

            cos = ttnn.reshape(cos, [1, position_ids.shape[0], 1, self.head_dim])  # [1, batch, 1, head_dim]
            sin = ttnn.reshape(sin, [1, position_ids.shape[0], 1, self.head_dim])  # [1, batch, 1, head_dim]

            cos = ttnn.to_layout(cos, ttnn.TILE_LAYOUT)
            sin = ttnn.to_layout(sin, ttnn.TILE_LAYOUT)

            grid = (
                ttnn.CoreRangeSet(ttnn.num_cores_to_corerange_set(self.batch, self.core_grid, row_wise=True))
                .bounding_box()
                .grid_size()
            )
            mem_config = ttnn.create_sharded_memory_config(
                shape=(1, self.batch, ttnn.TILE_SIZE, self.head_dim),
                core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )

            cos = ttnn.interleaved_to_sharded(
                cos, mem_config
            )  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
            sin = ttnn.interleaved_to_sharded(
                sin, mem_config
            )  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

            return cos, sin

    def forward(self, xq, xk, cos, sin):
        xq = self.apply_rotary(xq, cos, sin)
        xk = self.apply_rotary(xk, cos, sin)
        return xq, xk
