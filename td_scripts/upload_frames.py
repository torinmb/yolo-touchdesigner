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

def _pack_from_planar_mono(arr_planar, H, W3):
    """
    Fast path: input is shader-style planar mono with shape (H, 3*W, C>=1).
    We mirror your previous code: take .r channel, then slice width into 3 planes.
    Copy into preallocated CHW payload: [R...][G...][B...]
    """
    W = W3 // 3
    # If float, quantize like UNORM8; else ensure uint8
    if arr_planar.dtype == np.float32:
        mono_u8 = (arr_planar[..., 0] * 255.0 + 0.5).astype(np.uint8, copy=False)
    elif arr_planar.dtype == np.uint8:
        mono_u8 = arr_planar[..., 0]
    else:
        mono_u8 = np.clip(arr_planar[..., 0], 0, 255).astype(np.uint8, copy=False)

    r = mono_u8[:, 0*W:1*W].ravel(order='C')
    g = mono_u8[:, 1*W:2*W].ravel(order='C')
    b = mono_u8[:, 2*W:3*W].ravel(order='C')

    np.copyto(_payload[0*_HW:1*_HW], r, casting='no')
    np.copyto(_payload[1*_HW:2*_HW], g, casting='no')
    np.copyto(_payload[2*_HW:3*_HW], b, casting='no')

def _pack_from_interleaved_rgb(arr_rgb, H, W):
    """
    Fallback: CPU version of your compute shader.
    Shader logic:
      out(x, y) for 0..W-1   = R(x, y)
      out(x+W, y)            = G(x, y)
      out(x+2W, y)           = B(x, y)
    We do that implicitly by writing each channel into the correct CHW plane.
    This exactly matches the shader’s plane order & UNORM8 quantization.
    """
    # Ensure RGB
    if arr_rgb.shape[2] < 3:
        
        arr_rgb = np.repeat(arr_rgb, 3, axis=2)
    else:
        arr_rgb = arr_rgb[:, :, :3]

    # Quantize like the shader’s UNORM8 path
    if arr_rgb.dtype == np.float32:
        rgb_u8 = (arr_rgb * 255.0 + 0.5).astype(np.uint8, copy=False)
    elif arr_rgb.dtype == np.uint8:
        rgb_u8 = arr_rgb
    else:
        rgb_u8 = np.clip(arr_rgb, 0, 255).astype(np.uint8, copy=False)

    # Write each plane exactly as the shader would have produced then we’d slice it:
    # Instead of building an intermediate (H,3W) mono array, write directly to CHW.
    planes = _payload.reshape(3, INPUT_H, INPUT_W)  # views over _payload
    # R plane goes to plane 0
    np.copyto(planes[0], rgb_u8[:, :, 0], casting='no')
    # G plane goes to plane 1
    np.copyto(planes[1], rgb_u8[:, :, 1], casting='no')
    # B plane goes to plane 2
    np.copyto(planes[2], rgb_u8[:, :, 2], casting='no')

def send_frame_u8_chw():
    webserver = op('yolo_server/webserver1')

    # Flow Control: If browser is busy (CPU mode slow), skip frame to prevent TD stall
    if webserver.fetch('busy', False):
        client = op('yolo_server/active_client').text.strip()
        if client:
             # Send lightweight sync pulse (JSON) instead of heavy binary
             # This keeps op('tick') and frame sync logic alive in TD
             
             msg = json.dumps({"sync": True, "tick": absTime.frame, "frame": absTime.frame})
             webserver.webSocketSendText(client, msg)

        # Auto-reset if stuck for > 60 frames (1s)
        if absTime.frame - webserver.fetch('busy_ts', 0) < 60:
            # op('frame').text = absTime.frame
            return

    client    = op('yolo_server/active_client').text.strip()
    if not client:
        return
    
    top = op('source')  # Either compute output (H,3W,mono/RGBA) or direct 640x640 RGB
    if top is None:
        return

    # Mark busy immediately to prevent next frame from entering pipeline
    webserver.store('busy', True)
    webserver.store('busy_ts', absTime.frame)

    arr = top.numpyArray(delayed=True)  # HxWxC
    if arr is None:
        return

    H, W_or_W3, C = arr.shape

    # ---------- PATH A: Shader-style planar mono (H, 3*W, C>=1) ----------
    if (W_or_W3 % 3) == 0 and C >= 1:
        
        _pack_from_planar_mono(arr, H, W_or_W3)
        _pay_mv[:_payload.nbytes] = _payload_mv
        try:
            _send(client, webserver, _payload.nbytes, H, W_or_W3 // 3)
        except Exception as e:
            debug('webSocketSendBinary error (planar):', e)
        op('frame').text = absTime.frame
        return

    # ---------- PATH B: CPU fallback (expected 640x640 interleaved RGB) ----------
    # Strictly match your stated fallback dimensions to keep preallocations valid.
    if H != INPUT_H or W_or_W3 != INPUT_W:
        # Dimensions don’t match the preallocated buffer — skip to avoid format drift.
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
