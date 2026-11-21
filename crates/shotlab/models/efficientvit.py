# EfficientViT-seg B1 for ADE20K (Cai et al., ICCV 2023), written out after mit-han-lab/efficientvit's
# efficientvit/models (Apache-2.0) so that its checkpoint, loaded weights-only and strictly, exports to ONNX without
# running that repository (models/README). 6 x 3 x 384 x 384 in, 6 x 150 x 48 x 48 class logits out.
# usage: python efficientvit.py efficientvit_seg_b1_ade20k.pt out.onnx (torch 2.14 and onnx; then onnxslim)
import sys, torch, torch.nn as nn, torch.nn.functional as F

H = nn.Hardswish


class Conv(nn.Module):
    def __init__(s, i, o, k=3, st=1, g=1, bias=False, norm=True, act=None):
        super().__init__()
        s.conv = nn.Conv2d(i, o, k, st, k // 2, groups=g, bias=bias)
        s.norm = nn.BatchNorm2d(o) if norm else None
        s.act = act

    def forward(s, x):
        x = s.conv(x)
        x = s.norm(x) if s.norm is not None else x
        return s.act(x) if s.act is not None else x


class DS(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.depth_conv = Conv(c, c, 3, 1, c, act=H())
        s.point_conv = Conv(c, c, 1)

    def forward(s, x):
        return s.point_conv(s.depth_conv(x))


class MB(nn.Module):
    # fewer: the backbone's blocks before its attention stages' (biases, no norm but the last)
    def __init__(s, i, o, st, fewer):
        super().__init__()
        m = 4 * i
        s.inverted_conv = Conv(i, m, 1, bias=fewer, norm=not fewer, act=H())
        s.depth_conv = Conv(m, m, 3, st, m, bias=fewer, norm=not fewer, act=H())
        s.point_conv = Conv(m, o, 1)

    def forward(s, x):
        return s.point_conv(s.depth_conv(s.inverted_conv(x)))


class Res(nn.Module):
    def __init__(s, main, short=True):
        super().__init__()
        s.main, s.short = main, short

    def forward(s, x):
        return s.main(x) + x if s.short else s.main(x)


class Seq(nn.Module):
    def __init__(s, ops):
        super().__init__()
        s.op_list = nn.ModuleList(ops)

    def forward(s, x):
        for o in s.op_list:
            x = o(x)
        return x


class MLA(nn.Module):
    # lightweight multi-scale linear attention: ReLU kernels, the 5 x 5 aggregated q, k, v as heads of their own
    def __init__(s, c, dim=16):
        super().__init__()
        h, s.dim = c // dim, dim
        t = h * dim
        s.qkv = Conv(c, 3 * t, 1, norm=False)
        s.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3 * t, 3 * t, 5, 1, 2, groups=3 * t, bias=False),
                    nn.Conv2d(3 * t, 3 * t, 1, groups=3 * h, bias=False),
                )
            ]
        )
        s.proj = Conv(2 * t, c, 1)

    def forward(s, x):
        q = s.qkv(x)
        q = torch.cat([q, s.aggreg[0](q)], 1)
        b, _, h, w = q.shape
        q = q.reshape(b, -1, 3 * s.dim, h * w)
        a, k, v = F.relu(q[:, :, : s.dim]), F.relu(q[:, :, s.dim : 2 * s.dim]), q[:, :, 2 * s.dim :]
        o = torch.matmul(torch.matmul(F.pad(v, (0, 0, 0, 1), value=1.0), k.transpose(-1, -2)), a)
        return s.proj((o[:, :, :-1] / (o[:, :, -1:] + 1e-15)).reshape(b, -1, h, w))


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.context_module = Res(MLA(c))
        s.local_module = Res(MB(c, c, 1, True))

    def forward(s, x):
        return s.local_module(s.context_module(x))


class Backbone(nn.Module):
    def __init__(s, w=(16, 32, 64, 128, 256), d=(1, 2, 3, 3, 4)):
        super().__init__()
        s.input_stem = Seq([Conv(3, w[0], 3, 2, act=H())] + [Res(DS(w[0])) for _ in range(d[0])])
        st, c = [], w[0]
        for wi, di in zip(w[1:3], d[1:3]):
            st.append(Seq([Res(MB(c if i == 0 else wi, wi, 2 if i == 0 else 1, False), i > 0) for i in range(di)]))
            c = wi
        for wi, di in zip(w[3:], d[3:]):
            st.append(Seq([Res(MB(c, wi, 2, True), False)] + [Block(wi) for _ in range(di)]))
            c = wi
        s.stages = nn.ModuleList(st)

    def forward(s, x):
        f = [s.input_stem(x)]
        for t in s.stages:
            f.append(t(f[-1]))
        return f


class Head(nn.Module):
    # stages 4, 3, 2 to 64 channels at stride 8 (bicubic), summed, three MBConvs, the classifier
    def __init__(s, w=64, n=150):
        super().__init__()
        s.input_ops = nn.ModuleList([Seq([Conv(256, w, 1)]), Seq([Conv(128, w, 1)]), Conv(64, w, 1)])
        s.middle = Seq([Res(MB(w, w, 1, False)) for _ in range(3)])
        s.output_ops = nn.ModuleList([Seq([Conv(w, n, 1, bias=True, norm=False)])])

    def forward(s, f):
        up = lambda i, t, k: F.interpolate(s.input_ops[i](t), scale_factor=k, mode="bicubic", align_corners=False)
        return s.output_ops[0](s.middle(up(0, f[4], 4) + up(1, f[3], 2) + s.input_ops[2](f[2])))


class Seg(nn.Module):
    def __init__(s):
        super().__init__()
        s.backbone, s.head = Backbone(), Head()

    def forward(s, x):
        return s.head(s.backbone(x))


m = Seg()
m.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["state_dict"], strict=True)
m.eval()
x = torch.zeros(6, 3, 384, 384)
torch.onnx.export(m, (x,), sys.argv[2], input_names=["pixel_values"], output_names=["logits"], opset_version=17, dynamo=False)
