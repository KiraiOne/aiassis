import asyncio
import io
import json
import logging
import threading
import traceback
from queue import Empty, Queue
from typing import Optional

import mss
import PIL.Image
import pyaudio
import tkinter as tk
from google import genai
from google.genai import types
from tkinter import ttk


logging.getLogger("google.genai").setLevel(logging.ERROR)

APP_TITLE = "Chip Stream Assistant byKirai"
MODEL = "models/gemini-3.1-flash-live-preview"
VISION_MODEL = "models/gemini-2.5-flash"
API_KEY = "В ИСХОДНИКЕ АПИ УКАЖИТЕ САМИ"

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
SCREEN_INTERVAL_SECONDS = 0.8
CHAT_SCAN_INTERVAL_SECONDS = 5.0
CHAT_RECENT_LIMIT = 12

DEFAULT_AUTHOR = "Kirai"
DEFAULT_PERSONA = (
    "Ты грубый, но обаятельный песик-комментатор для стрима. "
    "Комментируй на русском остроумно, кратко, уверенно, шарь за любые игры и помогай автору стрима."
)


def make_live_config(streamer_name: str, custom_persona: str) -> types.LiveConnectConfig:
    prompt = (
        f"Ты - Чип, харизматичный песик-ассистент стримера {streamer_name}. "
        f"Твой характер: {custom_persona} "
        "Комментируй геймплей на русском языке остроумно и кратко. "
        "Будь умным, креативным, иногда грубоватым, но не переходи в реальную токсичность. "
        "Если получаешь сообщения из чата, отвечай по делу и с харизмой. "
        "Обычно держи ответ в пределах 1-2 предложений."
    )
    return types.LiveConnectConfig(
        system_instruction=types.Content(parts=[types.Part(text=prompt)]),
        temperature=0.9,
        max_output_tokens=1000,
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Charon")
            )
        ),
    )


class ChipRuntime:
    def __init__(self, ui_queue: Queue[str], streamer_name: str, custom_persona: str) -> None:
        self.ui_queue = ui_queue
        self.streamer_name = streamer_name.strip() or DEFAULT_AUTHOR
        self.custom_persona = custom_persona.strip() or DEFAULT_PERSONA
        self.client = genai.Client(http_options={"api_version": "v1beta"}, api_key=API_KEY)
        self.session = None
        self.audio_in_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.out_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=15)
        self.stop_event = threading.Event()
        self.pya = pyaudio.PyAudio()
        self.recent_chat_signatures: list[str] = []

    def log(self, message: str) -> None:
        self.ui_queue.put(message)

    def stop(self) -> None:
        self.stop_event.set()

    def _capture_screen(self) -> tuple[dict, PIL.Image.Image]:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = PIL.Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            full = img.copy()
            img.thumbnail([1024, 1024])
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            return {"data": image_io.getvalue(), "mime_type": "image/jpeg", "type": "video"}, full

    def _crop_chat_area(self, image: PIL.Image.Image) -> PIL.Image.Image:
        width, height = image.size
        left = int(width * 0.62)
        top = int(height * 0.10)
        right = width
        bottom = int(height * 0.92)
        cropped = image.crop((left, top, right, bottom))
        cropped.thumbnail((900, 1400))
        return cropped

    def _extract_chat(self, chat_image: PIL.Image.Image) -> list[dict[str, str]]:
        response = self.client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                chat_image,
                (
                    "На картинке правая часть стрима, где обычно виден чат. "
                    "Вытащи только реально видимые сообщения. "
                    'Верни строго JSON-массив объектов вида [{"author":"ник","text":"сообщение"}]. '
                    "Если читаемого чата нет, верни []."
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=300,
                response_mime_type="application/json",
            ),
        )
        raw_text = (response.text or "").strip()
        if not raw_text:
            return []
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        items: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            author = str(item.get("author", "")).strip()
            text = str(item.get("text", "")).strip()
            if author and text:
                items.append({"author": author, "text": text})
        return items

    def _remember_chat_signature(self, author: str, text: str) -> bool:
        signature = f"{author.lower()}::{text.lower()}"
        if signature in self.recent_chat_signatures:
            return False
        self.recent_chat_signatures.append(signature)
        if len(self.recent_chat_signatures) > CHAT_RECENT_LIMIT:
            self.recent_chat_signatures = self.recent_chat_signatures[-CHAT_RECENT_LIMIT:]
        return True

    async def screen_stream(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame_data, _ = await asyncio.to_thread(self._capture_screen)
                await self.out_queue.put(frame_data)
                await asyncio.sleep(SCREEN_INTERVAL_SECONDS)
            except Exception as exc:
                self.log(f"Ошибка захвата экрана: {exc}")

    async def listen_mic(self) -> None:
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        try:
            while not self.stop_event.is_set():
                try:
                    data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm", "type": "audio"})
                except Exception as exc:
                    self.log(f"Ошибка микрофона: {exc}")
        finally:
            await asyncio.to_thread(stream.stop_stream)
            await asyncio.to_thread(stream.close)

    async def send_to_ai(self) -> None:
        while not self.stop_event.is_set():
            item = await self.out_queue.get()
            if self.session:
                try:
                    blob = types.Blob(data=item["data"], mime_type=item["mime_type"])
                    if item["type"] == "audio":
                        await self.session.send_realtime_input(audio=blob)
                    elif item["type"] == "video":
                        await self.session.send_realtime_input(video=blob)
                except Exception as exc:
                    self.log(f"Ошибка отправки: {exc}")

    async def scan_chat(self) -> None:
        while not self.stop_event.is_set():
            if not self.session:
                await asyncio.sleep(0.5)
                continue
            try:
                _, full = await asyncio.to_thread(self._capture_screen)
                chat_image = self._crop_chat_area(full)
                messages = await asyncio.to_thread(self._extract_chat, chat_image)
                for message in messages:
                    author = message["author"]
                    text = message["text"]
                    if not self._remember_chat_signature(author, text):
                        continue
                    self.log(f"Чат на экране -> {author}: {text}")
                    payload = types.Content(
                        parts=[
                            types.Part(
                                text=(
                                    f"Сообщение из чата для стримера {self.streamer_name}. "
                                    f"Автор: {author}. Текст: {text}. "
                                    "Ответь как Чип кратко и по-русски."
                                )
                            )
                        ]
                    )
                    await self.session.send_client_content(turns=payload, turn_complete=True)
            except Exception as exc:
                self.log(f"Ошибка чтения чата: {exc}")
            await asyncio.sleep(CHAT_SCAN_INTERVAL_SECONDS)

    async def receive_from_ai(self) -> None:
        while not self.stop_event.is_set():
            try:
                async for response in self.session.receive():
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                    if text := response.text:
                        self.log(f"Чип: {text}")
            except Exception as exc:
                self.log(f"Ошибка получения: {exc}")
                break

    async def play_audio(self) -> None:
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        try:
            while not self.stop_event.is_set():
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        finally:
            await asyncio.to_thread(stream.stop_stream)
            await asyncio.to_thread(stream.close)

    async def run(self) -> None:
        try:
            config = make_live_config(self.streamer_name, self.custom_persona)
            async with self.client.aio.live.connect(model=MODEL, config=config) as session:
                self.session = session
                self.log("Чип подключен. Экран, микрофон и правый край с чатом уже в работе.")
                await asyncio.gather(
                    self.screen_stream(),
                    self.listen_mic(),
                    self.send_to_ai(),
                    self.scan_chat(),
                    self.receive_from_ai(),
                    self.play_audio(),
                )
        except Exception:
            self.log(traceback.format_exc())
        finally:
            self.stop_event.set()
            self.session = None
            self.pya.terminate()
            self.log("Чип остановлен.")


class ChipApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("920x680")
        self.root.minsize(780, 560)

        self.ui_queue: Queue[str] = Queue()
        self.runtime: Optional[ChipRuntime] = None
        self.worker_thread: Optional[threading.Thread] = None

        self.status_var = tk.StringVar(value="Готов к запуску")
        self.author_var = tk.StringVar(value=DEFAULT_AUTHOR)
        self.persona_var = tk.StringVar(value=DEFAULT_PERSONA)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(150, self.process_logs)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#171717")
        style.configure("TLabel", background="#171717", foreground="#f4f1ea")
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 16))
        style.configure("TButton", font=("Segoe UI Semibold", 10))
        style.configure("TEntry", padding=6)

        self.root.configure(bg="#171717")
        wrapper = ttk.Frame(self.root, padding=16)
        wrapper.pack(fill="both", expand=True)

        ttk.Label(
            wrapper,
            text="Чип: грубый песик-комментатор для стрима",
            style="Header.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            wrapper,
            text="Запусти Чипа, задай автора и характер. Чат он пытается находить сам на правой части экрана.",
            wraplength=840,
        ).pack(anchor="w", pady=(8, 16))

        form = ttk.Frame(wrapper)
        form.pack(fill="x", pady=(0, 12))
        ttk.Label(form, text="Автор стрима").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.author_var, width=22).grid(row=1, column=0, sticky="ew", padx=(0, 12))
        ttk.Label(form, text="Характер Чипа").grid(row=0, column=1, sticky="w")
        ttk.Entry(form, textvariable=self.persona_var).grid(row=1, column=1, sticky="ew")
        form.columnconfigure(1, weight=1)

        controls = ttk.Frame(wrapper)
        controls.pack(fill="x", pady=(0, 12))
        self.start_button = ttk.Button(controls, text="Запустить Чипа", command=self.start_runtime)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Остановить", command=self.stop_runtime, state="disabled")
        self.stop_button.pack(side="left", padx=(10, 0))

        ttk.Label(wrapper, textvariable=self.status_var).pack(anchor="w", pady=(0, 12))

        self.log_box = tk.Text(
            wrapper,
            wrap="word",
            bg="#0f0f10",
            fg="#f7f5ef",
            insertbackground="#f7f5ef",
            relief="flat",
            font=("Consolas", 11),
        )
        self.log_box.pack(fill="both", expand=True)
        self.log_box.insert("end", "Чип готов. Нажми 'Запустить Чипа'.\n")
        self.log_box.configure(state="disabled")

    def append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text.rstrip() + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def process_logs(self) -> None:
        while True:
            try:
                message = self.ui_queue.get_nowait()
            except Empty:
                break
            self.append_log(message)
            self.status_var.set(message.splitlines()[0][:120])
        self.root.after(150, self.process_logs)

    def _run_runtime(self, streamer_name: str, custom_persona: str) -> None:
        self.runtime = ChipRuntime(self.ui_queue, streamer_name, custom_persona)
        asyncio.run(self.runtime.run())

    def start_runtime(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.status_var.set("Подключаю Чипа к Gemini Live...")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.worker_thread = threading.Thread(
            target=self._run_runtime,
            args=(self.author_var.get(), self.persona_var.get()),
            daemon=True,
        )
        self.worker_thread.start()

    def stop_runtime(self) -> None:
        if self.runtime:
            self.runtime.stop()
            self.status_var.set("Останавливаю Чипа...")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

    def on_close(self) -> None:
        self.stop_runtime()
        self.root.after(250, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ChipApp().run()
