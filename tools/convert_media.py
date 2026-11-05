"""Convert the backup-RL result GIFs/PNGs and the paper's figures into web-friendly assets.

GIFs become cropped H.264 MP4s (yuv420p, faststart) plus a WebP poster of the
final frame; figures become WebP. Crops remove the renderer's white margins;
clips that are compared side by side share one crop so their scale matches.

Usage:
    python tools/convert_media.py --backup-rl /path/to/backup-RL \
        --arxiv-src /path/to/unzipped/arxiv/source --out assets [--only NAME ...]
Requires ffmpeg (libx264), mutool (to rasterize PDF figures), and Pillow with WebP.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageSequence

QUAD = 'results_analytics/objX_quadrotor_traces'
HERO = ('outputs_objF_quadBackup_ra_smooth_dLQR_new/'
        'quad_backup_ra-20260423_001127-objFraSmooth-id2-s2-b0.99-tc8.0-cc8.0-ts5000000-tau0.0025-es0.10-ec0.10-asw0.05-cs0.80-vp1.5-tp30.0-csm0.8/'
        'comparison/quadBackup_compare-20260423_011639-allRA-id2-kpz36p0-kvz24p0-kvxy14p0-attp45p0-yaw0p35-cm0p60-zsafe32p0-cvz18p0-lat1p50-minz0p0-traj')

# (output name, source GIF relative to backup-RL, crop x, y, w, h, output width)
QUAD_CROP = (480, 220, 864, 864)
VIDEOS = [
    ('quad_ps2_sa', f'{QUAD}/BLISS-SA/BLISS-SA_isometric.gif', *QUAD_CROP, 864),
    ('quad_ps2_abp', f'{QUAD}/BLISS-ABP/BLISS-ABP_isometric.gif', *QUAD_CROP, 864),
    ('quad_cbf_rl', f'{QUAD}/CBF-RL/CBF-RL_isometric.gif', *QUAD_CROP, 864),
    ('quad_mps', f'{QUAD}/MPS/MPS_isometric.gif', *QUAD_CROP, 864),
    ('quad_cpo', f'{QUAD}/CPO/CPO_isometric.gif', *QUAD_CROP, 864),
    ('quad_sac_lag', f'{QUAD}/SAC-Lag/SAC-Lag_isometric.gif', *QUAD_CROP, 864),
    ('quad_sac_pen_high', f'{QUAD}/SAC-PenHigh/SAC-PenHigh_isometric.gif', *QUAD_CROP, 864),
    ('quad_sac_pen_low', f'{QUAD}/SAC-PenLow/SAC-PenLow_isometric.gif', *QUAD_CROP, 864),
    ('quad_multisafety', 'results_analytics/quadrotor_iris_multiSafety/seed3_best_return_front.gif',
     480, 100, 860, 900, 860),
    ('phase1_learned', f'{HERO}/hero_gif-overall.gif', 1290, 60, 3000, 2760, 1400),
    ('phase1_analytic', f'{HERO}/hero_gif-analytic-overall.gif', 1290, 60, 3000, 2760, 1400),
]

# (output name, source, max width, render dpi, crop as width fractions (x0, x1) or None).
# 'src:' paths are relative to the arXiv source directory; others to backup-RL.
IMAGES = [
    ('fig1_full', 'src:figures/PS2_Main_Figure.pdf', 3000, 120, None),
    ('unicycle_sets', 'src:figures/unicycle_final.pdf', 2400, 200, (0.0, 0.7270)),     # Fig. 2(a)-(c)
    ('hj_comparison', 'src:figures/unicycle_final.pdf', 1100, 200, (0.7272, 1.0)),     # Fig. 2(d)
    ('ablation_altitude', 'src:figures/quadrotor_ablation.pdf', 1800, 600, None),      # Fig. 9
    ('unicycle_traces', 'results_analytics/objX_unicycle_traces/ieee_unicycle_traces.png', 1400, 0, None),
]


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def last_frame(gif):
    im = Image.open(gif)
    frame = None
    for frame in ImageSequence.Iterator(im):
        pass
    return frame.convert('RGB')


def video(src, dst_dir, name, x, y, w, h, out_w):
    mp4 = dst_dir / f'{name}.mp4'
    vf = f'crop={w}:{h}:{x}:{y},scale={out_w}:-2:flags=lanczos,format=yuv420p'
    run(['ffmpeg', '-y', '-v', 'error', '-i', str(src), '-vf', vf, '-c:v', 'libx264', '-preset', 'slow',
         '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(mp4)])
    poster = last_frame(src).crop((x, y, x + w, y + h))
    poster = poster.resize((out_w, round(h * out_w / w)), Image.LANCZOS)
    poster.save(dst_dir / f'{name}.webp', 'WEBP', quality=82, method=6)
    return mp4.stat().st_size


def rasterize(pdf, dpi):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'page.png'
        run(['mutool', 'draw', '-q', '-r', str(dpi), '-o', str(out), str(pdf), '1'])
        return Image.open(out).copy()


def trim(im, pad=12):
    """Crop uniform white margins, keeping `pad` pixels."""
    gray = im.convert('L').point(lambda v: 255 if v < 245 else 0)
    box = gray.getbbox()
    if not box:
        return im
    x0, y0, x1, y1 = box
    return im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))


def image(src, dst_dir, name, max_w, dpi, crop):
    im = rasterize(src, dpi) if src.suffix == '.pdf' else Image.open(src)
    im = im.convert('RGBA')
    flat = Image.new('RGBA', im.size, (255, 255, 255, 255))
    flat.alpha_composite(im)
    im = flat.convert('RGB')
    if crop:
        im = trim(im.crop((round(crop[0] * im.width), 0, round(crop[1] * im.width), im.height)))
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    out = dst_dir / f'{name}.webp'
    im.save(out, 'WEBP', quality=90, method=6)
    return out.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backup-rl', required=True, type=Path)
    ap.add_argument('--arxiv-src', required=True, type=Path, help='unzipped arXiv source (main.tex, figures/)')
    ap.add_argument('--out', default=Path('assets'), type=Path)
    ap.add_argument('--only', nargs='*', help='subset of output names')
    a = ap.parse_args()
    (a.out / 'video').mkdir(parents=True, exist_ok=True)
    (a.out / 'img').mkdir(parents=True, exist_ok=True)
    for name, rel, x, y, w, h, out_w in VIDEOS:
        if a.only and name not in a.only:
            continue
        size = video(a.backup_rl / rel, a.out / 'video', name, x, y, w, h, out_w)
        print(f'{name:22s} {size / 1e6:6.2f} MB')
    for name, src, max_w, dpi, crop in IMAGES:
        if a.only and name not in a.only:
            continue
        path = a.arxiv_src / src[4:] if src.startswith('src:') else a.backup_rl / src
        size = image(path, a.out / 'img', name, max_w, dpi, crop)
        print(f'{name:22s} {size / 1e6:6.2f} MB')


if __name__ == '__main__':
    main()
