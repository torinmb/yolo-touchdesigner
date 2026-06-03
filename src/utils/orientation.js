// Copyright (c) 2025 Blankensmithing LLC
// This file is licensed under the GNU Affero General Public License v3.0
// (or later), see https://github.com/torinmb/yolo-touchdesigner/blob/master/LICENSE.txt.

export function normalizeRotationDeg(value) {
    const deg = Number(value);
    if (!Number.isFinite(deg)) return 0;

    const normalized = ((Math.trunc(deg) % 360) + 360) % 360;
    if (normalized === 90 || normalized === 180 || normalized === 270) {
        return normalized;
    }
    return 0;
}

export function isQuarterTurn(rotationDeg) {
    const normalized = normalizeRotationDeg(rotationDeg);
    return normalized === 90 || normalized === 270;
}

export function getRotatedSize(width, height, rotationDeg) {
    return isQuarterTurn(rotationDeg)
        ? { width: height, height: width }
        : { width, height };
}

export function getOrientedSourceCoords(
    x,
    y,
    width,
    height,
    flipH = false,
    flipV = false,
    rotationDeg = 0,
) {
    const rotation = normalizeRotationDeg(rotationDeg);
    const rotated = getRotatedSize(width, height, rotation);

    let tx = flipH ? rotated.width - 1 - x : x;
    let ty = flipV ? rotated.height - 1 - y : y;

    switch (rotation) {
        case 90:
            return { x: ty, y: height - 1 - tx };
        case 180:
            return { x: width - 1 - tx, y: height - 1 - ty };
        case 270:
            return { x: width - 1 - ty, y: tx };
        default:
            return { x: tx, y: ty };
    }
}

export function formatCssTransform(
    flipH = false,
    flipV = false,
    rotationDeg = 0,
) {
    const transforms = [];
    if (flipH) transforms.push("scaleX(-1)");
    if (flipV) transforms.push("scaleY(-1)");
    if (rotationDeg)
        transforms.push(`rotate(${normalizeRotationDeg(rotationDeg)}deg)`);
    return transforms.length ? transforms.join(" ") : "none";
}
