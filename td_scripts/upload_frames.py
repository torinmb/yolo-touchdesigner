# Execute DAT
import numpy as np
import struct
import json
import time

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
    if arr_planar.dtype == np.uint8:
        # ZERO-COPY FAST PATH: If C=1 and it is C-contiguous, we can use the array directly
        # without allocating a new payload array or copying data!
        if arr_planar.ndim == 3 and arr_planar.shape[2] == 1:
            mono_u8 = arr_planar[:, :, 0]
            if mono_u8.flags['C_CONTIGUOUS']:
                return mono_u8
        mono_u8 = arr_planar[..., 0]
    elif arr_planar.dtype == np.float32:
        mono_u8 = (arr_planar[..., 0] * 255.0 + 0.5).astype(np.uint8, copy=False)
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

    # Super-fast 1D slice copy (bypasses expensive 3D strided transpose copies)
    _HW = H * W
    payload = np.empty(3 * _HW, dtype=np.uint8)
    payload[:_HW] = rgb_u8[:, :, 0].ravel()
    payload[_HW:2*_HW] = rgb_u8[:, :, 1].ravel()
    payload[2*_HW:] = rgb_u8[:, :, 2].ravel()
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

webserver = op('yolo_server/webserver1')
client_op = op('yolo_server/active_client')
top = op('source') 
def send_frame_u8_chw():
    
    # 1. Flow Control Check: If busy, skip frame to prevent network/pipeline flooding
    if webserver.fetch('busy', False):
        # Auto-reset busy timeout if stuck for > 1.0 second of real time
        if time.time() - webserver.fetch('busy_ts', 0.0) >= 1.0:
            debug("Warning: Pipeline busy timeout. Resetting busy flag.")
            webserver.store('busy', False)
        else:
            # Send sync tick if client is active to keep the link alive
            client = client_op.text.strip()
            if client:
                msg = json.dumps({"sync": True, "tick": absTime.frame, "frame": absTime.frame})
                webserver.webSocketSendText(client, msg)
            return

    client = client_op.text.strip()
    if not client:
        return
    
    
    if top is None:
        return

    # Fetch array asynchronously from GPU
    arr = top.numpyArray(delayed=True)  # HxWxC
    if arr is None:
        return

    # Deep copy the array on the main thread to ensure absolute thread-safety.
    # This prevents the background thread from accessing active TouchDesigner-managed GPU/CPU-mapped memory,
    # which can lead to tearing, race conditions, or segmentation faults during frame drops.
    arr_copy = arr.copy()

    # Mark busy immediately to lock the pipeline during packing/inference
    now = time.time()
    webserver.store('busy', True)
    webserver.store('busy_ts', now)

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
            _repack_task_fn(arr_copy, H, W_or_W3, C, result_container)
            if result_container['success']:
                _send_payload(client, webserver, result_container['payload'],
                              result_container['h_final'], result_container['w_final'], frame_num)
                op('frame').text = frame_num
                # Note: 'busy' remains True here as the browser will clear it upon receiving predictions
            else:
                debug("Synchronous repack failed:", result_container['error'])
                webserver.store('busy', False)
        except Exception as e:
            debug("Synchronous repack exception:", e)
            webserver.store('busy', False)
        return

    # Main thread callbacks
    frame_op = op('frame')
    def on_success(*args, **kwargs):
        try:
            sent = False
            if result_container['success']:
                payload = result_container['payload']
                h_final = result_container['h_final']
                w_final = result_container['w_final']
                
                # Check that client and webserver are still valid
                current_client = client_op
                if current_client and current_client.text.strip() == client:
                    _send_payload(client, webserver, payload, h_final, w_final, frame_num)
                    frame_op.text = frame_num
                    sent = True
            else:
                debug("Threaded repack failed:", result_container['error'])

            # If we didn't successfully send a frame payload, release the busy flag immediately
            if not sent:
                webserver.store('busy', False)

        except Exception as ex:
            debug("Error in Threaded SuccessHook:", ex)
            webserver.store('busy', False)

    def on_except(*args, **kwargs):
        debug("Exception occurred during threaded texture repack task.")
        webserver.store('busy', False)

    # Create the task
    task = TDTask(
        target=_repack_task_fn,
        args=(arr_copy, H, W_or_W3, C, result_container),
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
