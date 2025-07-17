import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from flamingo_pytorch import GatedCrossAttentionBlock, PerceiverResampler
# from flamingo_palm import Residual, ParallelTransformerBlock, LayerNorm
from flamingo_pytorch import Residual, ParallelTransformerBlock, LayerNorm



def exists(val):
    return val is not None

# transformer


class FlamingoMAPF(nn.Module):
    def __init__(
        self,
        gnn_dim, 
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        media_token_id=3,
        cross_attn_every=3,
        img_encoder=None,
        perceiver_num_latents=64,
        perceiver_depth=2,
        max_video_frames = None,
        only_attend_immediate_media=True
    ):
        super().__init__()

        # self.num_agents = num_agents
        self.action_emb = nn.Linear(5, dim)  
        self.gnn_emb = nn.Linear(gnn_dim, dim)  
        
        self.media_token_id = media_token_id 

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, only_attend_immediate_media=only_attend_immediate_media) if not (ind % cross_attn_every) else None
            ]))

        self.to_actions = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, 5, bias=False)
        )

    
    def forward(
        self,
        actions,
        gnn_embeddings
    ):

        actions = self.action_emb(actions)

        gnn_embeddings = self.gnn_emb(gnn_embeddings)

        media_locations = None



        for i, (attn_ff, flamingo_cross_attn) in enumerate(self.layers) :
            actions_tokens = attn_ff(actions)
            if exists(flamingo_cross_attn):
                actions_tokens = flamingo_cross_attn(
                    actions_tokens,
                    gnn_embeddings
                )



        
        return self.to_actions(actions_tokens)


