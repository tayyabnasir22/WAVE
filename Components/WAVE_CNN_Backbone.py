from Components.BoundaryRefinementBlock import BoundaryRefinementBlock
from Components.DownsampleX16 import DownsampleX16
from pytorch_wavelets import DWTForward
import torch.nn as nn
import torch
from Components.DinoV3.Dino3VisionTransformer import Dino3VisionTransformer
from Components.SToRA import SToRA
from Components.StructureBlock import StructureBlock
from Components.DetailBlock import DetailBlock
from Components.DetailInjector import DetailInjector
from Components.DBPUpsampling import DBPUpsampling

class WAVE(nn.Module):
    def __init__(
            self, 
            use_pretrained: bool, 
            num_feats: int = 64, 
            img_size: int = 256, 
            patch_size: int = 16, 
            scale: int = 4,
            kernel_size = 3,
            freeze_dino: bool = True
        ):
        super(WAVE, self).__init__()
        self.patch_size = patch_size

        # Init processing components
        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.init_depth_projection = DownsampleX16(1, num_feats)
        
        self.upsampler1 = DBPUpsampling(num_feats, num_feats*2, 2)
        self.upsampler2 = DBPUpsampling(num_feats*2, num_feats*3, 2)
        self.upsampler3 = DBPUpsampling(num_feats*3, num_feats*4, 2)

        self.final_upscale = DBPUpsampling(num_feats*4, num_feats*4, 2)

        # DINO compoenents
        self.intermediate_layer_idx = [2, 5, 9, 11]
        
        self.semantics_encoder = Dino3VisionTransformer(
            img_size=img_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            ffn_ratio=4,
            patch_size=self.patch_size,
            mask_k_bias=True,
            layerscale_init=1e-5,     
            n_storage_tokens=4,       
            ffn_layer="mlp", 
        )

        if use_pretrained == True:
            # To load from local disk
            # ckpt_path = PathManager.GetBasePath() + "dinov2_vits14_pretrain.pth"

            # ckpt = torch.load(ckpt_path, map_location="cpu")

            # To load from server
            ckpt = torch.hub.load_state_dict_from_url(
                "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNHpiOTJ3enZpNjFwYWF2cHp2cHFnODcyIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzY0OTY1MjB9fX1dfQ__&Signature=SjHrDsrDPKklzUzy5JUdtle8aLDtw9wkskyfT4M0702RunMey3Y5Jna61ewUYo-ynCdCpJ%7EIn2EHa1Sv7nuZj6-X-7lfTk2SqYq4Jxgk%7E85bwL2P2i3a-PLG393tGtFPWEdN3ovcW7ScIGrW8dM0XRj9iWqYA1SS2wUN3d71YKV2hjwIgi36qqnYGL9TAIcLeQzNvoT1spMLk-PYeUXuppOLa%7EzzXBT88DUR-oMZPthDR3FU8mBbaARJBa2IfF10F4CuNFtfx5m1YWseoCS%7EGRqBzpOGixS01UUgR6u%7Ed2Nx3gEHXeLP1Pi2kmVf%7E4538qS5MnrkT-qFYF7eOhZ6wA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=989805356906888",
                map_location="cpu"
            )

            filtered_state = {}
            for k, v in ckpt.items():
                if k.startswith("head") or k == "pos_embed":
                    continue
                filtered_state[k] = v

            missing, unexpected = self.semantics_encoder.load_state_dict(filtered_state, strict=False)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        if freeze_dino == True:
            # Freeze all DINO parameters
            for param in self.semantics_encoder.parameters():
                param.requires_grad = False

        self.stora1 = SToRA()
        self.stora2 = SToRA()
        self.stora3 = SToRA()
        self.stora4 = SToRA()


        # Wavelet compoenents
        self.wavelet_encoder = DWTForward(J=1, wave='haar')


        # detail blocks
        self.deatil_block1 = DetailBlock(384, num_feats)
        self.deatil_block2 = DetailBlock(384, num_feats*2)
        self.deatil_block3 = DetailBlock(384, num_feats*3)
        self.deatil_block4 = DetailBlock(384, num_feats*4)

        # structure blocks
        self.structure_block1 = StructureBlock(384, num_feats, kernel_size)
        self.structure_block2 = StructureBlock(384, num_feats*2, kernel_size)
        self.structure_block3 = StructureBlock(384, num_feats*3, kernel_size)
        self.structure_block4 = StructureBlock(384, num_feats*4, kernel_size)

        # Gated details injection
        self.injectors1 = DetailInjector(num_feats, kernel_size)
        self.injectors2 = DetailInjector(num_feats*2, kernel_size)
        self.injectors3 = DetailInjector(num_feats*3, kernel_size)
        self.injectors4 = DetailInjector(num_feats*4, kernel_size)


        self.residual1 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*2, out_channels=num_feats*2, kernel_size=1),
            nn.GroupNorm(8, num_feats*2),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*3, out_channels=num_feats*3, kernel_size=1),
            nn.GroupNorm(8, num_feats*3),
        )
        self.residual3 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*4, out_channels=num_feats*4, kernel_size=1),
            nn.GroupNorm(8, num_feats*4),
        )
        self.residual4 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*4 + num_feats, out_channels=num_feats*4, kernel_size=1),
            nn.GroupNorm(8, num_feats*4),
        )

        # Boundary refinmenet
        self.boundary_refiner = BoundaryRefinementBlock(3, num_feats)
                
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Conv2d(num_feats * 4, num_feats * 3, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats * 3, num_feats * 2, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats * 2, num_feats, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats, num_feats // 2, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats // 2, 1, kernel_size=3, bias=True, padding=1),
            
        )

    def dwt_full_pyramid(self, x, level=4, wave='haar'):
        LLs = []
        LHs = []
        HLs = []
        HHs = []
        current = x
        for i in range(level):
            yl, yh = self.wavelet_encoder(current)
            # Low-frequency
            LLs.append(yl) # Saving the input as LL not the output, and last output will be discarded for now TODO may need to change this

            # High-frequency bands
            y = yh[0]  # (B, C, 3, H, W)

            LHs.append(y[:, :, 0, :, :])
            HLs.append(y[:, :, 1, :, :])
            HHs.append(y[:, :, 2, :, :])

            # Next level input
            current = yl

        return LLs, LHs, HLs, HHs

    def forward(self, image, depth):
        H = image.shape[-2] 
        W = image.shape[-1] 
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        depth_bicubic = self.bicubic(depth)
        depth0 = self.init_depth_projection(depth_bicubic)

        features = self.semantics_encoder.get_intermediate_layers(image, n=self.intermediate_layer_idx, return_class_token=True)
                
        LLs, LHs, HLs, HHs = self.dwt_full_pyramid(image)

        # HUMMA Block 1        
        token1 = self.stora1(features[3][0], patch_h, patch_w)        
        details1 = self.deatil_block1(token1, LHs[3], HLs[3], HHs[3])
        depth1 = self.structure_block1(token1, LLs[3], depth0)
        depth1 = self.injectors1(depth1, details1)
        res1 = self.residual1(self.upsampler1(depth1))

        # HUMMA Block 2
        token2 = self.stora2(features[2][0], patch_h, patch_w)
        details2 = self.deatil_block2(token2, LHs[2], HLs[2], HHs[2])
        depth2 = self.structure_block2(token2, LLs[2], res1)
        depth2 = self.injectors2(depth2, details2)
        res2 = self.residual2(self.upsampler2(depth2))

        # HUMMA Block 3
        token3 = self.stora3(features[1][0], patch_h, patch_w)
        details3 = self.deatil_block3(token3, LHs[1], HLs[1], HHs[1])
        depth3 = self.structure_block3(token3, LLs[1], res2)
        depth3 = self.injectors3(depth3, details3)
        res3 = self.residual3(self.upsampler3(depth3))

        # HUMMA Block 4
        token4 = self.stora4(features[0][0], patch_h, patch_w)
        details4 = self.deatil_block4(token4, LHs[0], HLs[0], HHs[0])
        depth4 = self.structure_block4(token4, LLs[0], res3)
        depth4 = self.injectors4(depth4, details4)
        res4 = self.final_upscale(depth4)

        # Boundary refinement
        last_token = self.semantics_encoder.forward_features(image)['x_norm_patchtokens']
        B, N, C = last_token.shape

        last_token = last_token.permute(0,2,1).reshape(B, C, patch_h, patch_w)
        refined, _ = self.boundary_refiner(image, last_token, depth)
        res4 = self.residual4(torch.cat([res4, refined], dim=1))

        # Final projection
        out = self.final_projection(res4)
        return out + depth_bicubic