// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

export function iou(a, b) {
    const [ax, ay, aw, ah] = a,
        [bx, by, bw, bh] = b;
    const ax2 = ax + aw,
        ay2 = ay + ah,
        bx2 = bx + bw,
        by2 = by + bh;
    const ix = Math.max(ax, bx),
        iy = Math.max(ay, by);
    const ix2 = Math.min(ax2, bx2),
        iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix),
        ih = Math.max(0, iy2 - iy);
    const inter = iw * ih;
    const uni = aw * ah + bw * bh - inter + 1e-9;
    return inter / uni;
}

export function flipYKeypointsNorm(kpts, H) {
    if (!kpts) return kpts;
    return kpts.map((k) => ({ x: k.x / H, y: (H - k.y) / H, score: k.score }));
}

export function mapBoxYFlipNorm([x, y, w, h], H) {
    return [x / H, (H - (y + h)) / H, w / H, h / H];
}

export function mapAngleToBottomLeft(angleRad) {
    // Input angle is in image coords (+Y down); output coords are bottom-left origin => negate.
    return -angleRad;
}

export function boxCenter([x, y, w, h]) {
    return [x + 0.5 * w, y + 0.5 * h];
}

export function polygonFromXYWHR([x, y, w, h], angleRad) {
    const [cx, cy] = boxCenter([x, y, w, h]);
    const hw = 0.5 * w,
        hh = 0.5 * h;
    const c = Math.cos(angleRad),
        s = Math.sin(angleRad);
    // Four corners in image coords (+Y down)
    const pts = [
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
    ].map(([px, py]) => [cx + px * c - py * s, cy + px * s + py * c]);
    return pts; // [[x1,y1],...]
}

export function normPolyYFlip(pts, H) {
    return pts.map(([px, py]) => [px / H, (H - py) / H]);
}
