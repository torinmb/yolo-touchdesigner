# Execute DAT
import numpy as np
import struct
import time
import json

HEADER_BYTES = 16
TYPE_TENSOR  = 10
DTYPE_U8     = 1
LAYOUT_CHW   = 1

# ------ Configure your input size ------
INPUT_H = 640
INPUT_W = 640   # logical content width (not 3*W)

# ------ Reusable buffers ------
_HW = INPUT_H * INPUT_W
_payload = np.empty(3 * _HW, dtype=np.uint8)          # [R...][G...][B...]
_buf = bytearray(HEADER_BYTES + _payload.nbytes)
_hdr_mv = memoryview(_buf)[:HEADER_BYTES]
_pay_mv = memoryview(_buf)[HEADER_BYTES:]
_payload_mv = memoryview(_payload)                     # view over numpy bytes

def _send(client, webserver, payload_len, H, W):
    seq      = int(absTime.frame % (1 << 32))
    td_frame = int(absTime.frame)
    struct.pack_into('<BBBBHHII', _buf, 0,
                     TYPE_TENSOR, DTYPE_U8, LAYOUT_CHW, 0,
                     H, W, seq, td_frame)
    webserver.webSocketSendBinary(client, _buf[:HEADER_BYTES + payload_len])

def _pack_from_vertical_planar_mono(arr_planar, H3, W):
    """
    Super-fast path: input is shader-style vertically packed planar mono (3*H, W, C>=1).
    Since memory is row-major, this matches CHW natively!
    """
    H = H3 // 3
    if arr_planar.dtype == np.float32:
        mono_u8 = (arr_planar[..., 0] * 255.0 + 0.5).astype(np.uint8, copy=False)
    elif arr_planar.dtype == np.uint8:
        mono_u8 = arr_planar[..., 0]
    else:
        mono_u8 = np.clip(arr_planar[..., 0], 0, 255).astype(np.uint8, copy=False)

    # Reshape mono_u8 directly to match CHW planes inside the preallocated payload
    planes = _payload.reshape(3, H, W)
    np.copyto(planes, mono_u8.reshape(3, H, W), casting='no')

def _pack_from_interleaved_rgb(arr_rgb, H, W):
    """
    Fallback: CPU version of compute shader doing interleaved to planar (CHW).
    """
    if arr_rgb.shape[2] < 3:
        arr_rgb = np.repeat(arr_rgb, 3, axis=2)
    else:
        arr_rgb = arr_rgb[:, :, :3]

    if arr_rgb.dtype == np.float32:
        rgb_u8 = (arr_rgb * 255.0 + 0.5).astype(np.uint8, copy=False)
    elif arr_rgb.dtype == np.uint8:
        rgb_u8 = arr_rgb
    else:
        rgb_u8 = np.clip(arr_rgb, 0, 255).astype(np.uint8, copy=False)

    planes = _payload.reshape(3, H, W)
    np.copyto(planes, rgb_u8.transpose(2, 0, 1), casting='no')

def send_frame_u8_chw():
    webserver = op('yolo_server/webserver1')

    # Flow Control: If browser is busy, skip frame to prevent TD stall
    if webserver.fetch('busy', False):
        client = op('yolo_server/active_client').text.strip()
        if client:
            msg = json.dumps({"sync": True, "tick": absTime.frame, "frame": absTime.frame})
            webserver.webSocketSendText(client, msg)

        # Auto-reset if stuck for > 60 frames (1s)
        if absTime.frame - webserver.fetch('busy_ts', 0) < 60:
            return

    client    = op('yolo_server/active_client').text.strip()
    if not client:
        return
    
    top = op('source')  # Either compute output (3H,W,mono/RGBA) or direct 640x640 RGB
    if top is None:
        return

    # Mark busy immediately to prevent next frame from entering pipeline
    webserver.store('busy', True)
    webserver.store('busy_ts', absTime.frame)

    arr = top.numpyArray(delayed=True)  # HxWxC
    if arr is None:
        return

    H, W_or_W3, C = arr.shape

    # ---------- PATH A: Shader-style vertical planar mono (3*H, W, C>=1) ----------
    if (H % 3) == 0 and W_or_W3 == INPUT_W and C >= 1:
        _pack_from_vertical_planar_mono(arr, H, W_or_W3)
        _pay_mv[:_payload.nbytes] = _payload_mv
        try:
            _send(client, webserver, _payload.nbytes, H // 3, W_or_W3)
        except Exception as e:
            debug('webSocketSendBinary error (vertical planar):', e)
        op('frame').text = absTime.frame
        return

    # ---------- PATH B: CPU fallback (expected 640x640 interleaved RGB) ----------
    if H != INPUT_H or W_or_W3 != INPUT_W:
        return

    _pack_from_interleaved_rgb(arr, INPUT_H, INPUT_W)
    _pay_mv[:_payload.nbytes] = _payload_mv
    try:
        _send(client, webserver, _payload.nbytes, INPUT_H, INPUT_W)
    except Exception as e:
        debug('webSocketSendBinary error (fallback):', e)

    op('frame').text = absTime.frame

# Example hook — trigger on a CHOP pulse/toggle or every frame:
def onValueChange(channel, sampleIndex, val, prev):
    if val != 0:
        send_frame_u8_chw()
    return
