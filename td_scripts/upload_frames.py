# Execute DAT
import numpy as np
import struct
import json

HEADER_BYTES = 16
TYPE_TENSOR  = 10
DTYPE_U8     = 1
LAYOUT_CHW   = 1

# ------ Configure your input size ------
INPUT_H = 640
INPUT_W = 640   # logical content width (not 3*W)

# Helper to dynamically extract the TDTask class from the system Thread Manager
def _get_td_task_class():
    try:
        tm = op.TDResources.ThreadManager
        ext_inst = None
        if hasattr(tm, 'ext'):
            if hasattr(tm.ext, 'ThreadManagerExt'):
                ext_inst = tm.ext.ThreadManagerExt
            elif len(tm.ext) > 0:
                ext_inst = tm.ext[0]
        
        if ext_inst is not None:
            if hasattr(ext_inst, 'TDTask'):
                return ext_inst.TDTask
            elif hasattr(ext_inst.__class__, 'TDTask'):
                return ext_inst.__class__.TDTask
            else:
                import sys
                mod = sys.modules.get(ext_inst.__class__.__module__)
                if mod and hasattr(mod, 'TDTask'):
                    return mod.TDTask
    except Exception as e:
        debug("Error finding TDTask class:", e)
    return None

def _pack_from_vertical_planar_mono(arr_planar, H3, W):
    """
    Thread-safe path: input is shader-style vertically packed planar mono (3*H, W, C>=1).
    Since memory is row-major, this matches CHW natively!
    """
    H = H3 // 3
    if arr_planar.dtype == np.float32:
        mono_u8 = (arr_planar[..., 0] * 255.0 + 0.5).astype(np.uint8, copy=False)
    elif arr_planar.dtype == np.uint8:
        mono_u8 = arr_planar[..., 0]
    else:
        mono_u8 = np.clip(arr_planar[..., 0], 0, 255).astype(np.uint8, copy=False)

    payload = np.empty(3 * H * W, dtype=np.uint8)
    planes = payload.reshape(3, H, W)
    np.copyto(planes, mono_u8.reshape(3, H, W), casting='no')
    return payload

def _pack_from_interleaved_rgb(arr_rgb, H, W):
    """
    Thread-safe fallback: CPU version of compute shader doing interleaved to planar (CHW).
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

    payload = np.empty(3 * H * W, dtype=np.uint8)
    planes = payload.reshape(3, H, W)
    np.copyto(planes, rgb_u8.transpose(2, 0, 1), casting='no')
    return payload

def _repack_task_fn(arr, H, W_or_W3, C, result_container):
    """
    Runs completely in the background thread.
    No TouchDesigner APIs (op, absTime, etc.) should be accessed here.
    """
    try:
        # ---------- PATH A: Shader-style vertical planar mono (3*H, W, C>=1) ----------
        if (H % 3) == 0 and W_or_W3 == INPUT_W and C >= 1:
            payload = _pack_from_vertical_planar_mono(arr, H, W_or_W3)
            result_container['payload'] = payload
            result_container['h_final'] = H // 3
            result_container['w_final'] = W_or_W3
            result_container['success'] = True
            return

        # ---------- PATH B: CPU fallback (expected 640x640 interleaved RGB) ----------
        if H == INPUT_H and W_or_W3 == INPUT_W:
            payload = _pack_from_interleaved_rgb(arr, INPUT_H, INPUT_W)
            result_container['payload'] = payload
            result_container['h_final'] = INPUT_H
            result_container['w_final'] = INPUT_W
            result_container['success'] = True
            return

        # Unsupported shape
        result_container['success'] = False
        result_container['error'] = "Unsupported array shape: {}".format(arr.shape)

    except Exception as e:
        result_container['success'] = False
        result_container['error'] = str(e)

def _send_payload(client, webserver, payload, H, W, frame_num):
    seq = int(frame_num % (1 << 32))
    buf = bytearray(HEADER_BYTES + payload.nbytes)
    struct.pack_into('<BBBBHHII', buf, 0,
                     TYPE_TENSOR, DTYPE_U8, LAYOUT_CHW, 0,
                     H, W, seq, frame_num)
    # Copy payload using memoryview for maximum efficiency (no extra copy)
    buf[HEADER_BYTES:] = memoryview(payload)
    webserver.webSocketSendBinary(client, buf)

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

    client = op('yolo_server/active_client').text.strip()
    if not client:
        return
    
    top = op('source')  # Either compute output (3H,W,mono/RGBA) or direct 640x640 RGB
    if top is None:
        return

    # Fetch array asynchronously
    arr = top.numpyArray(delayed=True)  # HxWxC
    if arr is None:
        return

    # Mark busy immediately to prevent next frame from entering pipeline
    webserver.store('busy', True)
    webserver.store('busy_ts', absTime.frame)

    H, W_or_W3, C = arr.shape
    frame_num = int(absTime.frame)

    # Initialize a result container to collect data from the thread
    result_container = {
        'success': False,
        'payload': None,
        'h_final': 0,
        'w_final': 0,
        'error': None
    }

    # Retrieve TDTask class dynamically
    TDTask = _get_td_task_class()
    if TDTask is None:
        debug("ThreadManager TDTask class could not be resolved! Falling back to synchronous packing.")
        # Synchronous fallback if Thread Manager is missing
        try:
            _repack_task_fn(arr, H, W_or_W3, C, result_container)
            if result_container['success']:
                _send_payload(client, webserver, result_container['payload'],
                              result_container['h_final'], result_container['w_final'], frame_num)
                op('frame').text = frame_num
            else:
                debug("Synchronous repack failed:", result_container['error'])
        finally:
            webserver.store('busy', False)
        return

    # Main thread callbacks
    def on_success(*args, **kwargs):
        try:
            if result_container['success']:
                payload = result_container['payload']
                h_final = result_container['h_final']
                w_final = result_container['w_final']
                
                # Check that client and webserver are still valid
                current_client = op('yolo_server/active_client')
                if current_client and current_client.text.strip() == client:
                    _send_payload(client, webserver, payload, h_final, w_final, frame_num)
                    op('frame').text = frame_num
            else:
                debug("Threaded repack failed:", result_container['error'])
        except Exception as ex:
            debug("Error in Threaded SuccessHook:", ex)
        finally:
            webserver.store('busy', False)

    def on_except(*args, **kwargs):
        debug("Exception occurred during threaded texture repack task.")
        webserver.store('busy', False)

    # Create the task
    task = TDTask(
        target=_repack_task_fn,
        args=(arr, H, W_or_W3, C, result_container),
        SuccessHook=on_success,
        ExceptHook=on_except
    )

    # Enqueue task to the Thread Manager
    try:
        op.TDResources.ThreadManager.EnqueueTask(task)
    except Exception as e:
        debug("Failed to enqueue task to ThreadManager:", e)
        webserver.store('busy', False)

# Example hook — trigger on a CHOP pulse/toggle or every frame:
def onValueChange(channel, sampleIndex, val, prev):
    if val != 0:
        send_frame_u8_chw()
    return
