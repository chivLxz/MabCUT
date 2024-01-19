import torch
import torch.nn.functional as F
# global attention extractor
def global_extractor(self, feat_id, feat, feats, num_patches=64, patch_ids=None, attn_mats=None):
    B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
    feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
    if num_patches > 0:
        if feat_id < 3:
            if patch_ids is not None:
                patch_id = patch_ids[feat_id]
            else:
                patch_id = torch.randperm(feat_reshape.shape[1],
                                          device=feats[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            attn_qs = torch.zeros(1).to(feat.device)
        else:
            if attn_mats is not None:
                attn_qs = attn_mats[feat_id]
            else:
                feat_q = feat_reshape
                feat_k = feat_reshape.permute(0, 2, 1)
                dots = torch.bmm(feat_q, feat_k)
                attn = dots.softmax(dim=2)
                prob = -torch.log(attn)
                prob = torch.where(torch.isinf(prob), torch.full_like(prob, 0), prob)
                entropy = torch.sum(torch.mul(attn, prob), dim=2)
                _, index = torch.sort(entropy)
                patch_id = index[:, :num_patches]
                attn_qs = attn[torch.arange(B)[:, None], patch_id, :]
            feat_reshape = torch.bmm(attn_qs, feat_reshape)
            x_sample = feat_reshape.flatten(0, 1)
            patch_id = []
    else:
        x_sample = feat_reshape
        patch_id = []
    if self.use_mlp:
        mlp = getattr(self, 'mlp_%d' % feat_id)
        x_sample = mlp(x_sample)
    x_sample = self.l2norm(x_sample)

    if num_patches == 0:
        x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
    return x_sample, patch_id, attn_qs

# local attention extractor
def local_extractor(self, feat_id, feat, feats, num_patches=64, patch_ids=None, attn_mats=None):
    k_s = 7
    B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
    feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
    if num_patches > 0:
        if feat_id < 3:
            if patch_ids is not None:
                patch_id = patch_ids[feat_id]
            else:
                patch_id = torch.randperm(feat_reshape.shape[1],
                                          device=feats[0].device)
                patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            attn_qs = torch.zeros(1).to(feat.device)
        else:
            feat_local = F.unfold(feat, kernel_size=k_s, stride=1, padding=3)  # (B, ks*ks*C, L)
            L = feat_local.shape[2]
            if attn_mats is not None:
                patch_id = patch_ids[feat_id]
                attn_qs = attn_mats[feat_id]
            else:
                feat_k = feat_local.permute(0, 2, 1).reshape(B, L, k_s * k_s, C).flatten(0,
                                                                                         1)
                feat_q = feat_reshape.reshape(B * L, C, 1)
                dots_local = torch.bmm(feat_k, feat_q)
                attn_local = dots_local.softmax(dim=1)
                attn_local = attn_local.reshape(B, L, -1)
                prob = -torch.log(attn_local)
                prob = torch.where(torch.isinf(prob), torch.full_like(prob, 0), prob)
                entropy = torch.sum(torch.mul(attn_local, prob), dim=2)
                _, index = torch.sort(entropy)
                patch_id = index[:, :num_patches]
                attn_qs = attn_local[torch.arange(B)[:, None], patch_id, :]
                attn_qs = attn_qs.flatten(0, 1).unsqueeze(1)
            feat_v = feat_local[torch.arange(B)[:, None], :, patch_id].permute(0, 2, 1)
            feat_v = feat_v.flatten(0, 1).view(B * num_patches, k_s * k_s, C)
            feat_reshape = torch.bmm(attn_qs, feat_v)
            x_sample = feat_reshape.flatten(0, 1)
    else:
        x_sample = feat_reshape
        patch_id = []
    if self.use_mlp:
        mlp = getattr(self, 'mlp_%d' % feat_id)
        x_sample = mlp(x_sample)
    x_sample = self.l2norm(x_sample)

    if num_patches == 0:
        x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
    return x_sample, patch_id, attn_qs
