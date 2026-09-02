"""Generate a signed induced-flow-rate field around heat pump wells.

Replaces the sparse delta-function well channel with the analytic induced
Darcy velocity field of point wells in a confined aquifer:

    v_r(r) = Q / (2*pi*b*r)   radially outward for injection (+Q),
                              inward for extraction (-Q)

superposed over all wells, using the local aquifer thickness b.
The induced velocity components REPLACE the initial Darcy velocity
channels (indices 3 and 4), so inputs stay [6, H, W]:
    0 conductivity, 1 hydraulic head, 2 aquifer thickness,
    3 induced velocity x, 4 induced velocity y, 5 well flow rates.
Optionally adds a drawdown head channel: dh = sum Q/(2*pi*K*b) * ln(r_ref/r).

Usage:
    python generate_wellflow_field.py \
        --src step3_predV_residual_exclude11_13 \
        --dest step3_predV_residual_exclude11_13_wf
"""

import argparse
import math
import shutil
from pathlib import Path

import torch
import yaml

WELL_TOL = 1.0


def denorm(t, spec):
    return t * (spec["max"] - spec["min"]) + spec["min"]


def compute_fields(inp, info, cell, vfloor, vmax, rmin_cells, with_head):
    q = denorm(inp[info["Inputs"]["Wells: Max Flow Rates [m^3/d]"]["index"]],
               info["Inputs"]["Wells: Max Flow Rates [m^3/d]"])
    thick = info["Inputs"]["Aquifer Thickness (t0) [m]"]
    b = denorm(inp[thick["index"]], thick)
    cond = info["Inputs"]["Conductivity [m/d]"]
    k = denorm(inp[cond["index"]], cond) if with_head else None

    ys, xs = torch.where(q.abs() > WELL_TOL)
    h, w = q.shape[-2:]
    rmin = cell * rmin_cells
    vx = torch.zeros(h, w, dtype=torch.float32)
    vy = torch.zeros(h, w, dtype=torch.float32)
    dh = torch.zeros(h, w, dtype=torch.float32) if with_head else None
    wells = []

    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        qw = float(-q[y0, x0])
        bw = max(float(b[y0, x0]), 0.05)
        wells.append((x0, y0, qw))
        coef = abs(qw) / (2.0 * math.pi * bw)
        rmax_px = coef / vfloor / cell
        i0 = max(int(y0 - rmax_px) - 1, 0)
        i1 = min(int(y0 + rmax_px) + 2, h)
        j0 = max(int(x0 - rmax_px) - 1, 0)
        j1 = min(int(x0 + rmax_px) + 2, w)

        dy = (torch.arange(i0, i1, dtype=torch.float32) - y0) * cell
        dx = (torch.arange(j0, j1, dtype=torch.float32) - x0) * cell
        DX = dx.unsqueeze(0).expand(dy.numel(), j1 - j0)
        DY = dy.unsqueeze(1).expand(i1 - i0, dx.numel())
        r2 = (DX * DX + DY * DY).clamp_min(rmin * rmin)
        s = qw / (2.0 * math.pi * bw) / r2
        vx[i0:i1, j0:j1] += s * DX
        vy[i0:i1, j0:j1] += s * DY
        if with_head:
            kw = k[i0:i1, j0:j1].clamp_min(1.0)
            dh[i0:i1, j0:j1] += qw / (2.0 * math.pi * kw * bw) * torch.log(
                torch.tensor(rmin) / r2.sqrt())

    speed = (vx * vx + vy * vy).sqrt()
    over = speed > vmax
    if over.any():
        scale = torch.where(over, vmax / speed.clamp_min(1e-12), torch.ones_like(speed))
        vx = vx * scale
        vy = vy * scale
    return vy, vx, dh, wells

def generate_wellflow_field(src, dest=None, vfloor=5e-3, vmax=50.0, rmin_cells=0.5, with_head=False, replace_v=False, limit=None):

    dest = dest or Path(f"{str(src)}_wf{'_replace_v' if replace_v else '_not_replace_v'}")
    info = yaml.load(open(src / "info.yaml"), Loader=yaml.FullLoader)
    cell_x, cell_y = info["CellsSize"]
    assert math.isclose(cell_x, cell_y, rel_tol=1e-3), f"script assumes square cells, but cells are {cell_x} and {cell_y}"
    cell = float(cell_x)

    sims = sorted((src / "Inputs").glob("Sim_*.pt"))
    if limit:
        sims = sims[: limit]
    print(f"{len(sims)} sims -> {dest}")

    fields = ["Wells: Induced Darcy Velocity x [m/d]",
              "Wells: Induced Darcy Velocity y [m/d]"]
    if with_head:
        fields.append("Wells: Drawdown head [m]")
    stats = {}
    for sim in sims:
        inp = torch.load(sim)
        vx, vy, dh, wells = compute_fields(
            inp, info, cell, vfloor, vmax, rmin_cells, with_head)
        qsum = sum(w[2] for w in wells)
        n_inj = sum(w[2] > 0 for w in wells)
        fs = [vx.min().item(), vx.max().item(), vy.min().item(), vy.max().item()]
        if dh is not None:
            fs += [dh.min().item(), dh.max().item()]
        stats[sim.name] = fs
        print(f"  {sim.stem}: {len(wells)} wells ({n_inj} inj, {len(wells)-n_inj} ext) in {inp.shape[1:]} cells, "
              f"net Q={qsum:+.1f} m3/d, |v|max={max(-fs[0], fs[1], -fs[2], fs[3]):.1f}")

    gmins = [min(s[2 * i] for s in stats.values()) for i in range(len(fields))]
    gmaxs = [max(s[2 * i + 1] for s in stats.values()) for i in range(len(fields))]

    new_info = {"CellsSize": info["CellsSize"], "Inputs": dict(info["Inputs"]),
                "Labels": info["Labels"]}
    if replace_v:
        for old in ("Darcy Velocity in x (t0) [m/d]", "Darcy Velocity in y (t0) [m/d]"):
            new_info["Inputs"].pop(old)
        new_info["Inputs"][fields[0]] = {"index": 3, "min": gmins[0], "max": gmaxs[0]}
        new_info["Inputs"][fields[1]] = {"index": 4, "min": gmins[1], "max": gmaxs[1]}
    else:
        new_info["Inputs"][fields[0]] = {"index": 6, "min": gmins[0], "max": gmaxs[0]}
        new_info["Inputs"][fields[1]] = {"index": 7, "min": gmins[1], "max": gmaxs[1]}
    if with_head:
        new_info["Inputs"][fields[2]] = {"index": len(new_info["Inputs"]),
                                         "min": gmins[2], "max": gmaxs[2]}
    print("new channels:", {n: (round(v["min"], 3), round(v["max"], 3))
                            for n, v in new_info["Inputs"].items() if n not in info["Inputs"]})

    (dest / "Inputs").mkdir(parents=True, exist_ok=True)
    (dest / "Labels").mkdir(parents=True, exist_ok=True)
    for sim in sims:
        inp = torch.load(sim)
        vx, vy, dh, _ = compute_fields(
            inp, info, cell, vfloor, vmax, rmin_cells, with_head)
        chans = []
        for f, name in zip([vx, vy] + ([dh] if dh is not None else []), fields):
            spec = new_info["Inputs"][name]
            chans.append(((f - spec["min"]) / (spec["max"] - spec["min"])).unsqueeze(0))
        if replace_v:
            out = torch.cat([inp[:3], chans[0], chans[1], inp[5:6]] + chans[2:], dim=0).to(torch.float32)
        else:
            out = torch.cat((inp, *chans), dim=0).to(torch.float32)
        torch.save(out, dest / "Inputs" / sim.name)
        shutil.copy2(src / "Labels" / sim.name, dest / "Labels" / sim.name)
    yaml.dump(new_info, open(dest / "info.yaml", "w"))
    print(f"done -> {dest}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("step3_predV_residual_exclude11_13"))
    p.add_argument("--dest", type=Path, default=None)
    p.add_argument("--vfloor", type=float, default=5e-3, help="cutoff [m/d]; well influence ignored below this")
    p.add_argument("--vmax", type=float, default=50.0, help="clip induced speed to this [m/d]")
    p.add_argument("--rmin-cells", type=float, default=0.5, help="well radius in cells (singularity guard)")
    p.add_argument("--with-head", action="store_true", help="add a drawdown head channel")
    p.add_argument("--replace-v", action="store_true", help="replace input channels of v0 or simply append inputs")
    p.add_argument("--limit", type=int, default=None, help="process first N sims only")
    args = p.parse_args()

    generate_wellflow_field(
        src=args.src,
        dest=args.dest,
        vfloor=args.vfloor,
        vmax=args.vmax,
        rmin_cells=args.rmin_cells,
        with_head=args.with_head,
        replace_v=args.replace_v,
        limit=args.limit,
    )  

if __name__ == "__main__":
    main()
