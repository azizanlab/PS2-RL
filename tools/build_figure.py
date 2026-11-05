"""Build the scroll-animation assets for Figure 1 from the Figma layer export.

Inputs
  --layers     directory of the Figma PNG export (main_figure.zip, unzipped)
  --positions  tools/figure_layout.json: layer placements in 2x-frame pixels plus paint order
  --reference  the flattened Figure 1 render (3x), used for the baked "learned blob" layer
Outputs
  assets/figure/layers/<id>.png and assets/js/figure-manifest.js (layers in paint order)

To refresh the art after editing the figure in Figma, re-export the layers with the
same names and re-run this script; positions only need re-locating if layers moved.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

# Figma layer name (without .png) -> stable id used by assets/js/scrolly.js
IDS = {
    'Group 67': 'bg', 'PS2-RL Framework': 'title', 'Arrow 39': 'big-arrow',
    'Rectangle 37': 'p1-panel', 'Phase I_ Solves Subproblem 2': 'p1-title',
    'Rectangle 38': 'p2-panel', 'Phase II_ Solves Subproblem 1': 'p2-title',
    'Rectangle 39': 'obj1-box', 'Group 65': 'obj1',
    'Rectangle 42': 'base-panel', 'Base Set & Controller': 'base-title', 'Ellipse 76': 'base-B',
    'mathcal B-1': 'base-B-label', 'Line 54': 'base-curve', 'Line 55': 'base-dash',
    'Line 57': 'base-arrow-l', 'Line 56': 'base-arrow-m', 'Line 58': 'base-arrow-r',
    'Ellipse 77': 'base-xstar-dot', 'x^/star': 'base-xstar',
    'mathcal B =/{ e^/top P e/le c_{/mathcal B}/}': 'base-eq-B',
    'pi_{/mathcal B} = u^/star -Ke': 'base-eq-pi', 'e=x-x^/star': 'base-eq-e',
    'Rectangle 34': 'sa-panel', 'Safe-Arrival Policy Training': 'sa-title', 'Rectangle 35': 'sa-band',
    'Q_{/mathrm{SA},/beta}(x,u)=/mathbb E/left[/sum_{k=0}^/infty/beta^k/left(/prod_{i=0}^{k-1}/mathbf{c}(x_i)/right)/mathbf b(x_k)/right]': 'sa-q',
    'Group 58': 'sa-indicators',
    'Vector 117': 'sa-F', 'Vector 118': 'sa-S', 'Ellipse 78': 'sa-B',
    'mathcal F': 'sa-F-label', 'mathcal S-1': 'sa-S-label', 'mathcal B-2': 'sa-B-label',
    'Arrow 41': 'sa-fast', 'Arrow 42': 'sa-slow', 'Arrow 44': 'sa-unsafe',
    'Group 50': 'sa-x-fast1', 'Group 51': 'sa-x-fast2',
    'Group 52': 'sa-x-slow1', 'Group 53': 'sa-x-slow2', 'Group 54': 'sa-x-slow3', 'Group 55': 'sa-x-slow4',
    'Star 4': 'sa-star-fast', 'Star 5': 'sa-star-slow',
    'Group 56': 'sa-leg-fast', 'text{fast, safe arrival}': 'sa-leg-fast-txt', 'Q_{/mathrm{SA},/beta}=/beta^3': 'sa-leg-fast-q',
    'Group 57': 'sa-leg-slow', 'text{slow, safe arrival}': 'sa-leg-slow-txt', 'Q_{/mathrm{SA},/beta}=/beta^5': 'sa-leg-slow-q',
    'Arrow 43': 'sa-leg-unsafe', 'text{unsafe}': 'sa-leg-unsafe-txt', 'Q_{/mathrm{SA},/beta}=0': 'sa-leg-unsafe-q',
    'Rectangle 36': 'learn-band',
    'text{Learn }/pi_{/mathrm{SA}}^/theta/text{ to maximize }Q_{/mathrm{SA},/beta}': 'learn-title',
    'Group 49': 'learn-nn',
    'Rectangle 20': 'bk-panel', 'Backup Policy': 'bk-title',
    'pi_b^/theta =/begin{cases}/pi_{/mathcal B},&x/in/mathcal B/pi_{/mathrm{SA}}^/theta, & x/notin/mathcal B/end{cases}': 'bk-eq',
    'Rectangle 22': 'bk-hl-B', 'Rectangle 21': 'bk-hl-SA', '(Linear) Base Controller': 'bk-base-txt',
    'Vector 77': 'bk-base-arrow', '(Frozen) Safe Arrival Policy': 'bk-sa-txt', 'Group 18': 'bk-nn', 'Vector 76': 'bk-nn-arrow',
    'Rectangle 40': 'obj2-box',
    'Objective_ Train a performant and provably safe RL policy through the Control-Invariant Layer': 'obj2',
    'Rectangle 41': 'maxj-box',
    'begin{align}_/max_{/pi_/phi}&/,J(/pi_/phi)/_/mathrm{s.t.}&/,/pi_/phi(x)/in/mathbb K_{/mathrm{BCBF}}/_&/,/forall x/in/mathcal C_T(/pi_b^/theta)_/end{align}': 'maxj',
    'Rectangle 33': 'cil-panel', 'Control-Invariant Layer': 'cil-title', 'Rectangle 33-1': 'cil-eq-box',
    'begin{align}_&/mathcal P_{/mathrm{CIL}}(/pi_/phi)(x) =/arg/min_u/_ u-/pi_/phi(x)/__2^2/_&/mathrm{s.t.}/,/text{stay where }/pi_b^/theta/text{ can bring to }/mathcal B__/end{align}': 'cil-eq',
    'Arrow 59': 'cil-leg-check', 'text{Check safe flow to }/mathcal B': 'cil-leg-check-txt',
    'Arrow 60': 'cil-leg-safe', 'text{Safe flow to }/mathcal B/text{ exists}': 'cil-leg-safe-txt',
    'Arrow 61': 'cil-leg-nosafe', 'text{No safe flow to }/mathcal B': 'cil-leg-nosafe-txt',
    'Vector 85': 'cil-S', 'Vector 86': 'cil-CT', 'Ellipse 49': 'cil-B', 'mathcal B': 'cil-B-label',
    'mathcal C_{T}': 'cil-CT-label', 'mathcal S': 'cil-S-label', 'Vector 120': 'cil-traj',
    'Vector 106': 'cil-flow1', 'Vector 109': 'cil-flow23', 'Vector 107': 'cil-flow4', 'Star 6': 'cil-star',
    'Arrow 58': 'cil-escape', 'Ellipse 83': 'cil-escape-dot', 'Vector 108': 'cil-noflow',
    'Rectangle 43': 'loop-box', 'Group 59': 'loop', 'pi_/phi': 'loop-pi', 'Vector 119': 'cil-wedge',
}
RIGHT_BLOB_X = 1300            # learn-row items right of this belong to the trained (right) blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', required=True, type=Path)
    ap.add_argument('--positions', default=Path('tools/figure_layout.json'), type=Path)
    ap.add_argument('--reference', required=True, type=Path)
    ap.add_argument('--site', default=Path('.'), type=Path)
    a = ap.parse_args()

    pos = json.load(open(a.positions))
    W, H = pos['frame_w'], pos['frame_h']
    zorder = pos['zorder']
    by = {l['name'][:-4]: l for l in pos['layers']}

    out_dir = a.site / 'assets' / 'figure' / 'layers'
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # learn-row items: blobs, small B ellipses, dots, arrows
    blobs = sorted(['Vector 97', 'Vector 116'], key=lambda n: by[n]['x'])
    smallB = sorted(['Ellipse 52', 'Ellipse 64'], key=lambda n: by[n]['x'])
    dots = [n for n in by if re.fullmatch(r'Ellipse (\d+)', n) and 53 <= int(n.split()[1]) <= 75 and n != 'Ellipse 64']
    left_dots = sorted([n for n in dots if by[n]['x'] < RIGHT_BLOB_X], key=lambda n: (by[n]['x'], by[n]['y']))
    small_arrows = [f'Arrow {i}' for i in range(47, 58)]
    big_arrows = sorted(['Arrow 45', 'Arrow 46'], key=lambda n: by[n]['x'])
    nodes = sorted([f'Ellipse {i}' for i in range(79, 83)], key=lambda n: by[n]['x'])
    ids = dict(IDS)
    ids[blobs[0]] = 'learn-blob-l'
    ids[smallB[0]] = 'learn-B-l'
    for i, n in enumerate(left_dots, 1):
        ids[n] = f'learn-dot{i}'
    ids[big_arrows[0]] = 'learn-arrow1'
    ids[big_arrows[1]] = 'learn-arrow2'
    for i, n in enumerate(nodes, 1):
        ids[n] = f'cil-node{i}'
    # the trained (right) blob is baked from the reference render: blob outline as mask, 3x pixels
    baked = {blobs[1], smallB[1], *small_arrows, *[n for n in dots if by[n]['x'] >= RIGHT_BLOB_X]}

    ref = cv2.imread(str(a.reference), cv2.IMREAD_UNCHANGED)
    bl = by[blobs[1]]
    m = 2
    bx, by_, bw, bh = bl['x'] - m, bl['y'] - m, bl['w'] + 2 * m, bl['h'] + 2 * m
    s = ref.shape[1] / W
    X0, Y0, X1, Y1 = round(bx * s), round(by_ * s), round((bx + bw) * s), round((by_ + bh) * s)
    crop = ref[Y0:Y1, X0:X1].copy()
    blob = cv2.imread(str(a.layers / f'{blobs[1]}.png'), cv2.IMREAD_UNCHANGED)
    alpha = np.zeros((bh, bw), np.uint8)
    alpha[m:m + bl['h'], m:m + bl['w']] = blob[..., 3]
    alpha = cv2.resize(alpha, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
    crop[..., 3] = np.minimum(crop[..., 3], alpha)
    cv2.imwrite(str(out_dir / 'learn-blob-r.png'), crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    layers = []
    for fn in zorder:
        n = fn[:-4]
        if n in baked:
            if n == blobs[1]:
                layers.append({'id': 'learn-blob-r', 'src': 'learn-blob-r.png', 'x': bx, 'y': by_, 'w': bw, 'h': bh,
                               'figma': 'baked: ' + n})
            continue
        if n not in ids:
            raise SystemExit(f'no id for layer {n!r}')
        l = by[n]
        shutil.copyfile(a.layers / fn, out_dir / f'{ids[n]}.png')
        layers.append({'id': ids[n], 'src': f'{ids[n]}.png', 'x': l['x'], 'y': l['y'], 'w': l['w'], 'h': l['h'],
                       'figma': n})
    assert len({l['id'] for l in layers}) == len(layers)
    manifest = {'width': W, 'height': H, 'base': 'assets/figure/layers/', 'layers': layers}
    js = ('/* Generated by tools/build_figure.py: Figure 1 layers in paint order, positions in\n'
          '   2x-export pixels of the Figma frame. Do not edit by hand. */\n'
          'window.PS2_FIGURE = ' + json.dumps(manifest, separators=(',', ':')) + ';\n')
    (a.site / 'assets' / 'js' / 'figure-manifest.js').write_text(js)

    # verification composite straight from the manifest
    canvas = np.full((H, W, 3), 255, np.float32)
    for l in layers:
        im = cv2.imread(str(out_dir / l['src']), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        im = cv2.resize(im, (l['w'], l['h']), interpolation=cv2.INTER_AREA)
        x, y = l['x'], l['y']
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + l['w']), min(H, y + l['h'])
        sub = im[y0 - y:y1 - y, x0 - x:x1 - x]
        al = sub[..., 3:4]
        canvas[y0:y1, x0:x1] = sub[..., :3] * 255 * al + canvas[y0:y1, x0:x1] * (1 - al)
    cv2.imwrite(str(a.site / 'tools' / 'manifest_check.png'), canvas.astype(np.uint8))
    print(f'{len(layers)} layers written')


if __name__ == '__main__':
    main()
