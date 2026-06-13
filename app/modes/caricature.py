import base64
import io
import os
import threading
import time
import logging

import cv2
import numpy as np
import PIL.Image
import requests

import app.services.text as text_module
from app.core.mode_manager import ModeManager


logger = logging.getLogger(__name__)


class Caricature:
    STATE_COUNTDOWN = 'countdown'
    STATE_CAPTURING = 'capturing'
    STATE_LOADING = 'loading'
    STATE_DISPLAYING = 'displaying'
    STATE_ERROR = 'error'

    COUNTDOWN_DURATION = 3  # seconds of countdown before capture
    DISPLAY_DURATION = 30   # seconds before auto-returning to clock

    def __init__(self, width, height, mode_manager):
        self.width = width
        self.height = height
        self.mode_manager = mode_manager
        self._lock = threading.Lock()
        self._generation = 0
        self._last_mode_start_time = None
        self.state = None
        self.caricature_dots = None
        self.countdown_start = None
        self.display_start_time = None
        self.error_msg = None

    def _reset(self):
        with self._lock:
            self._generation += 1
            self.state = self.STATE_COUNTDOWN
            self.caricature_dots = None
            self.countdown_start = time.time()
            self.display_start_time = None
            self.error_msg = None

    def _call_pixellab_api(self, image_b64, generation):
        try:
            api_token = os.getenv('PIXELLAB_API_TOKEN') or os.getenv('PIXELLAB_API_KEY')
            if not api_token:
                raise ValueError("Missing PIXELLAB_API_TOKEN")

            headers = {"Authorization": f"Bearer {api_token}"}

            # Encode reference image (camera frame)
            img_bytes = base64.b64decode(image_b64)
            src_pil = PIL.Image.open(io.BytesIO(img_bytes)).convert('RGB')
            src_w, src_h = src_pil.size
            ref_w, ref_h = self._fit_within_limit(src_w, src_h, 1024)
            if (ref_w, ref_h) != (src_w, src_h):
                src_pil = src_pil.resize((ref_w, ref_h), PIL.Image.LANCZOS)
            ref_buf = io.BytesIO()
            src_pil.save(ref_buf, format='JPEG', quality=85)
            ref_b64 = base64.b64encode(ref_buf.getvalue()).decode('utf-8')

            payload = {
                "description": "Create a monochromatic (1bit only black and white, no grey!) caricature of that person! The background should be black.",
                "image_size": {"width": self.width, "height": self.height},
                "no_background": False,
                "reference_images": [
                    {
                        "image": {"type": "base64", "base64": ref_b64, "format": "jpeg"},
                        "size": {"width": ref_w, "height": ref_h},
                    }
                ],
            }

            resp = requests.post(
                "https://api.pixellab.ai/v2/generate-image-v2",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if not resp.ok:
                logger.error("Caricature API generate-image-v2 error status=%s body=%s", resp.status_code, resp.text[:2000])
            resp.raise_for_status()

            job_id = resp.json()["background_job_id"]
            logger.info("Caricature job queued id=%s", job_id)

            # Poll until complete (max 2 minutes)
            result_pil = None
            deadline = time.time() + 120
            while time.time() < deadline:
                time.sleep(3)
                poll = requests.get(
                    f"https://api.pixellab.ai/v2/background-jobs/{job_id}",
                    headers=headers,
                    timeout=15,
                )
                poll.raise_for_status()
                data = poll.json()
                status = data.get("status")
                if status == "completed":
                    last = data.get("last_response") or {}
                    logger.info("Caricature job completed id=%s", job_id)
                    logger.debug("Caricature last_response keys=%s", list(last.keys()))
                    images = last.get("images", [])
                    if not images:
                        raise ValueError(f"No images in completed job: {last}")
                    logger.info("Caricature received %s image candidate(s)", len(images))
                    pil_images = []
                    for i, img_entry in enumerate(images):
                        img_b64_raw = img_entry.get("base64", "")
                        if img_b64_raw.startswith("data:"):
                            img_b64_raw = img_b64_raw.split(",", 1)[1]
                        pil = PIL.Image.open(io.BytesIO(base64.b64decode(img_b64_raw)))
                        pil.save(f'caricature_result_raw_{i}.png')
                        pil_images.append(pil)
                    best = self._pick_best_image(pil_images, ref_b64)
                    result_pil = pil_images[best]
                    break
                elif status == "failed":
                    raise ValueError(f"Job failed: {data.get('last_response')}")
                logger.debug("Caricature poll id=%s status=%s", job_id, status)

            if result_pil is None:
                raise TimeoutError("Caricature job timed out after 2 minutes")

            logger.info("Caricature result ready size=%s mode=%s", result_pil.size, result_pil.mode)

            dots = self._pil_image_to_dots(result_pil)

            with self._lock:
                if self._generation == generation:
                    self.caricature_dots = dots
                    self.state = self.STATE_DISPLAYING
                    self.display_start_time = time.time()
                    cv2.imwrite('caricature_result.png', self.caricature_dots * 255)
                    logger.info("Saved caricature result to caricature_result.png")

        except Exception:
            logger.exception("Caricature API workflow failed")
            with self._lock:
                if self._generation == generation:
                    self.state = self.STATE_ERROR
                    self.error_msg = "Generation failed"

    def _fit_within_limit(self, width, height, limit):
        max_side = max(width, height)
        if max_side <= limit:
            return width, height
        scale = limit / float(max_side)
        new_w = max(16, int(round(width * scale)))
        new_h = max(16, int(round(height * scale)))
        return new_w, new_h

    def _pil_image_to_dots(self, pil_image):
        # Flatten alpha onto white background before any conversion
        if pil_image.mode in ('RGBA', 'LA') or (pil_image.mode == 'P' and 'transparency' in pil_image.info):
            bg = PIL.Image.new('RGB', pil_image.size, (255, 255, 255))
            bg.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
            pil_image = bg
        # Resize then dither to 1-bit (Floyd-Steinberg) — better than hard-threshold for coloured pixel art
        resized = pil_image.convert('L').resize((self.width, self.height), PIL.Image.LANCZOS)
        dithered = resized.convert('1')  # PIL default: Floyd-Steinberg dithering
        # PIL '1' mode: 0=black, 255=white per pixel; map to 1=on (white), 0=off (black)
        arr = np.array(dithered, dtype=np.uint8)
        return (arr != 0).astype(np.uint8)

    def _pick_best_image(self, pil_images, original_b64):
        """Send candidate images + original photo to an AI; return the index of the best match."""
        if len(pil_images) <= 1:
            return 0

        # Scale up for AI visibility (nearest-neighbour preserves pixel art edges)
        scale = 4
        thumbs_b64 = []
        for img in pil_images:
            bw = img.convert('L').point(lambda p: 255 if p >= 128 else 0)
            scaled = bw.resize((bw.width * scale, bw.height * scale), PIL.Image.NEAREST)
            buf = io.BytesIO()
            scaled.save(buf, format='PNG')
            thumbs_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

        prompt = (
            f"The first image is the original photograph of a person. "
            f"The following {len(pil_images)} images are small pixel art caricature candidates "
            f"(each {pil_images[0].width}\u00d7{pil_images[0].height}px, shown scaled up {scale}x) "
            f"for a 28\u00d728 1-bit flip-dot display. "
            f"Choose the candidate in which the person from the photo is most clearly recognisable. "
            f"Reply with ONLY the 0-based index number of the best candidate (0 to {len(pil_images) - 1})."
        )

        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        try:
            if anthropic_key:
                return self._pick_with_anthropic(thumbs_b64, original_b64, prompt, anthropic_key, len(pil_images))
            elif openai_key:
                return self._pick_with_openai(thumbs_b64, original_b64, prompt, openai_key, len(pil_images))
            else:
                logger.info("No AI key available for image selection; using candidate index 0")
                return 0
        except Exception:
            logger.exception("AI image selection failed; using candidate index 0")
            return 0

    def _pick_with_anthropic(self, thumbs_b64, original_b64, prompt, api_key, n):
        content = [
            {"type": "text", "text": "Original photograph:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": original_b64}},
        ]
        for i, b64 in enumerate(thumbs_b64):
            content.append({"type": "text", "text": f"Candidate {i}:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        content.append({"type": "text", "text": prompt})
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        idx = int(text.split()[0])
        idx = max(0, min(idx, n - 1))
        logger.info("Anthropic chose image index=%s", idx)
        return idx

    def _pick_with_openai(self, thumbs_b64, original_b64, prompt, api_key, n):
        content = [
            {"type": "text", "text": "Original photograph:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_b64}", "detail": "low"}},
        ]
        for i, b64 in enumerate(thumbs_b64):
            content.append({"type": "text", "text": f"Candidate {i}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"},
            })
        content.append({"type": "text", "text": prompt})
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        idx = int(text.split()[0])
        idx = max(0, min(idx, n - 1))
        logger.info("OpenAI chose image index=%s", idx)
        return idx

    def _start_capture(self, camera_frame):
        if camera_frame is None:
            with self._lock:
                self.state = self.STATE_ERROR
                self.error_msg = "No frame"
            return

        success, buf = cv2.imencode('.jpg', camera_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            with self._lock:
                self.state = self.STATE_ERROR
                self.error_msg = "Encode failed"
            return

        cv2.imwrite('caricature_debug.jpg', camera_frame)
        logger.info("Saved caricature input capture to caricature_debug.jpg")

        image_b64 = base64.standard_b64encode(buf).decode('utf-8')

        with self._lock:
            generation = self._generation
            self.state = self.STATE_LOADING

        thread = threading.Thread(
            target=self._call_pixellab_api,
            args=(image_b64, generation),
            daemon=True,
        )
        thread.start()

    def get_frame(self, camera_frame):
        # Auto-reset when the mode is freshly entered
        if self._last_mode_start_time != self.mode_manager.mode_start_time:
            self._last_mode_start_time = self.mode_manager.mode_start_time
            self._reset()

        with self._lock:
            state = self.state
            countdown_start = self.countdown_start

        if state == self.STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start
            seconds_left = self.COUNTDOWN_DURATION - int(elapsed)
            if elapsed >= self.COUNTDOWN_DURATION:
                with self._lock:
                    self.state = self.STATE_CAPTURING
            return self._make_countdown_frame(max(1, seconds_left))

        if state == self.STATE_CAPTURING:
            self._start_capture(camera_frame)
            return self._make_loading_frame()

        if state == self.STATE_LOADING:
            return self._make_loading_frame()

        if state == self.STATE_DISPLAYING:
            if time.time() - self.display_start_time > self.DISPLAY_DURATION:
                self.mode_manager.set_mode(ModeManager.MODE_CLOCK)
                return np.zeros((self.height, self.width), dtype=np.uint8)
            with self._lock:
                return self.caricature_dots.copy()

        if state == self.STATE_ERROR:
            frame = np.zeros((self.height, self.width), dtype=np.uint8)
            text_module.write(frame, "ERR", x=1, y=10, size=5, style="regular")
            return frame

        return np.zeros((self.height, self.width), dtype=np.uint8)

    def _make_countdown_frame(self, seconds_left):
        """Show the countdown digit centred and scaled up 3x."""
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        # Render digit into a 3x5 buffer then scale 3x → 9x15
        small = np.zeros((5, 3), dtype=np.uint8)
        text_module.write(small, str(seconds_left), x=0, y=0, size=5, style="regular")
        big = np.kron(small, np.ones((3, 3), dtype=np.uint8))
        y_off = (self.height - big.shape[0]) // 2
        x_off = (self.width - big.shape[1]) // 2
        frame[y_off:y_off + big.shape[0], x_off:x_off + big.shape[1]] = big
        return frame

    def _make_loading_frame(self):
        """Ping-pong vertical bar sweeping left-right to indicate processing."""
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        # One full left-right-left cycle every 1.2 seconds
        cycle = (time.time() % 1.2) / 1.2  # 0.0 → 1.0
        pos = cycle * 2  # 0.0 → 2.0
        if pos > 1:
            pos = 2 - pos  # bounce back
        col = int(pos * (self.width - 2))
        frame[:, col:col + 2] = 1
        return frame
