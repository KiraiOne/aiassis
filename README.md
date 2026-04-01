# 🎮 Создание Игрового ИИ-Ассистента (Gemini Live API)

Этот проект превращает **Gemini 3.1 Flash** в живого комментатора вашего геймплея. ИИ видит ваш экран, слышит ваш микрофон и отвечает голосом в реальном времени.

---

## 🛠 Шаг 1: Полная установка "с нуля"

### 1.1. Установка Python
1. Скачайте Python 3.11+ с [python.org](https://www.python.org/downloads/).
2. **Обязательно** при установке поставьте галочку **"Add Python to PATH"**.
3. Проверьте установку, открыв терминал (`Win+R` -> `cmd`) и введя: `python --version`.

### 1.2. Получение ключа (API Key)
1. Зайдите в [Google AI Studio](https://aistudio.google.com/).
2. Нажмите **Get API Key** -> **Create API Key in new project**.
3. Сохраните ключ. **Внимание:** Бесплатный тариф имеет лимиты (около 3-15 минут активной трансляции видео, затем нужна пауза).

### 1.3. Установка библиотек
Откройте терминал и выполните:
```bash
pip install google-genai opencv-python pyaudio pillow mss
```
*Если `pyaudio` выдает ошибку, используйте: `pip install pipwin` и затем `pipwin install pyaudio`.*

---

## ⚙️ Шаг 2: Настройка ИИ (Голос, Роль, Лимиты)

В коде за поведение ИИ отвечает блок `CONFIG`. Вот как его настроить "под себя":

### 🗣 Выбор голоса (Voice Config)
Доступные пресеты (выбирайте в зависимости от жанра игры):
* **Puck** — Юморной, быстрый, энергичный (идеально для шутеров/экшена).
* **Charon** — Глубокий, спокойный, философский (для RPG/хорроров).
* **Kore** — Женский, четкий, рассудительный.
* **Zephyr** — Дружелюбный, стандартный ассистент.
* **Aoede** — Мягкий и вежливый.

### 🧠 Системная инструкция (Роль)
Это "характер" вашего бота. Впишите туда правила поведения.
* **Пример для Twitch-стиля:** *"Ты — стример. Комментируй игру остроумно, используй сленг, реагируй на смерти игрока сарказмом."*
* **Пример для Помощника:** *"Ты — навигатор. Подсказывай, куда идти, анализируй интерфейс игры (HP, патроны) и предупреждай об опасности."*

### 📉 Лимиты и Оптимизация (Generation Config)
* **Temperature (0.0 - 1.0):** Ставьте `1.0` для безумных шуток или `0.2` для серьезных советов.
* **Max Output Tokens:** Ограничьте до `100-150`, чтобы ИИ не болтал слишком долго, перебивая звуки игры.
* **Media Resolution:** `MEDIUM` — золотая середина. `HIGH` заставит ИИ видеть мелкий текст, но быстрее съест лимит.

---

## 📝 Шаг 3: Готовый код (Полная версия)

Создайте файл `game_ai.py` и вставьте этот код. **Не забудьте вставить свой API Key в строку 30.**

```python
import os
import asyncio
import base64
import io
import cv2
import pyaudio
import PIL.Image
import mss
import traceback
import logging
from google import genai
from google.genai import types

# Убираем лишние технические предупреждения в консоли, чтобы видеть только суть
logging.getLogger("google.genai").setLevel(logging.ERROR)

# --- [БЛОК 1: НАСТРОЙКИ ЗВУКА] ---
# Эти параметры согласуют работу твоего микрофона и динамиков с требованиями ИИ
FORMAT = pyaudio.paInt16        # Глубина звука 16 бит (стандартное качество)
CHANNELS = 1                    # Моно-канал (ИИ не нужно стерео для понимания речи)
SEND_SAMPLE_RATE = 16000        # Частота записи микрофона (16 кГц — идеал для распознавания речи)
RECEIVE_SAMPLE_RATE = 24000     # Частота воспроизведения ответа ИИ (24 кГц — чтобы голос был чистым)
CHUNK_SIZE = 1024               # Размер "кусочка" аудио. Чем меньше, тем быстрее реакция, но выше нагрузка

# --- [БЛОК 2: НАСТРОЙКИ ИИ] ---
MODEL = "models/gemini-3.1-flash-live-preview" # Самая быстрая модель с поддержкой видео-потока
API_KEY = "ВАШ_КЛЮЧ_ЗДЕСЬ" 

# Создаем "клиента" — это наш мост к серверам Google
client = genai.Client(http_options={"api_version": "v1beta"}, api_key=API_KEY)

# Конфигурация поведения: здесь мы описываем "личность" бота
CONFIG = types.LiveConnectConfig(
    system_instruction=types.Content(
        parts=[types.Part(text="""Ты — харизматичный игровой комментатор. 
        Ты видишь экран игрока. Твоя задача: остроумно комментировать геймплей на русском языке. 
        Будь эмоциональным, но кратким (не более 2 предложений за раз).""")]
    ),
    temperature=0.9,           # Уровень "фантазии": 0.9 делает ответы живыми и разными
    max_output_tokens=150,     # Ограничение длины ответа, чтобы бот не тараторил слишком долго
    response_modalities=["AUDIO"], # Указываем, что хотим получать ответ именно голосом
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck") # Имя голоса
        )
    ),
)

pya = pyaudio.PyAudio() # Инициализация звуковой системы Windows

class GameAssistant:
    def __init__(self):
        self.session = None
        self.audio_in_queue = asyncio.Queue()  # Очередь для хранения голоса от ИИ перед проигрыванием
        self.out_queue = asyncio.Queue(maxsize=15) # Очередь для отправки данных (экран + мик) к ИИ

    def _get_screen(self):
        """Функция захвата изображения с монитора"""
        with mss.mss() as sct:
            monitor = sct.monitors[1] # Индекс 1 — это твой главный монитор
            sct_img = sct.grab(monitor)
            # Превращаем скриншот в формат, который понимает библиотека обработки изображений
            img = PIL.Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            # Сжимаем картинку до 1024 пикселей, чтобы не перегружать интернет-канал
            img.thumbnail([1024, 1024])
            
            # Сохраняем результат в оперативную память как JPEG
            image_io = io.BytesIO()
            img.save(image_io, format="jpeg")
            # Возвращаем словарь с данными и меткой "video" для дальнейшей сортировки
            return {"data": image_io.getvalue(), "mime_type": "image/jpeg", "type": "video"}

    async def screen_stream(self):
        """Фоновая задача: делает скриншоты каждые 0.8 сек"""
        while True:
            try:
                frame_data = await asyncio.to_thread(self._get_screen)
                await self.out_queue.put(frame_data) # Кладем в очередь на отправку
                await asyncio.sleep(0.8) # Пауза, чтобы не превысить лимиты API (токены в минуту)
            except Exception as e:
                print(f"Ошибка захвата экрана: {e}")

    async def listen_mic(self):
        """Фоновая задача: постоянно слушает твой микрофон"""
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE, 
            input=True, frames_per_buffer=CHUNK_SIZE
        )
        while True:
            try:
                # Читаем байты с микрофона
                data = await asyncio.to_thread(stream.read, CHUNK_SIZE, exception_on_overflow=False)
                # Помечаем данные как "audio" и кладем в очередь
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm", "type": "audio"})
            except Exception as e:
                print(f"Ошибка микрофона: {e}")

    async def send_to_ai(self):
        """Фоновая задача: отправляет накопленные данные из очереди в Google"""
        while True:
            item = await self.out_queue.get()
            if self.session:
                try:
                    # Упаковываем байты в специальный контейнер Blob
                    blob = types.Blob(data=item["data"], mime_type=item["mime_type"])
                    # Отправляем в зависимости от типа (audio или video)
                    if item["type"] == "audio":
                        await self.session.send_realtime_input(audio=blob)
                    elif item["type"] == "video":
                        await self.session.send_realtime_input(video=blob)
                except Exception as e:
                    print(f"Ошибка отправки данных: {e}")

    async def receive_from_ai(self):
        """Фоновая задача: ловит ответы от ИИ"""
        while True:
            try:
                # Слушаем поток ответов от сессии
                async for response in self.session.receive():
                    if data := response.data:
                        # Если пришло аудио (голос ИИ) — кидаем в очередь на проигрывание
                        self.audio_in_queue.put_nowait(data)
                    if text := response.text:
                        # Если пришел текст — выводим в консоль для истории
                        print(f"AI: {text}")
            except Exception as e:
                print(f"Ошибка получения ответа: {e}")
                break

    async def play_audio(self):
        """Фоновая задача: выводит голос ИИ в твои динамики"""
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, 
            rate=RECEIVE_SAMPLE_RATE, output=True
        )
        while True:
            # Берем байты из очереди (которые пришли от ИИ)
            bytestream = await self.audio_in_queue.get()
            # Физически "играем" звук через звуковую карту
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        """Главная функция запуска всей системы"""
        try:
            # Устанавливаем соединение с сервером Gemini
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print("✅ Всё готово! Соединение активно. Попробуй заговорить или запусти игру.")
                # Запускаем все процессы ОДНОВРЕМЕННО
                await asyncio.gather(
                    self.screen_stream(), 
                    self.listen_mic(), 
                    self.send_to_ai(), 
                    self.receive_from_ai(), 
                    self.play_audio()
                )
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    # Точка входа: запускаем асинхронный цикл
    try:
        asyncio.run(GameAssistant().run())
    except KeyboardInterrupt:
        print("\n🛑 Программа остановлена.")
```

---

## 🚀 Шаг 4: Запуск игры
1. Запустите игру в режиме **"Окно без рамки"** (Borderless), чтобы захват экрана работал стабильнее.
2. Запустите скрипт: `python game_ai.py`.
3. Начните играть и говорить — ИИ увидит происходящее и ответит вам.

---



---

### Как это улучшить потом?
1. **Переводчик:** Если игра на английском, попросите ИИ в системной инструкции переводить важные сюжетные диалоги.
2. **Анализ логов:** Можно добавить чтение текстовых файлов игры (логов), чтобы ИИ знал точную статистику.
3. **OpenCV:** Можно добавить распознавание конкретных объектов (например, врагов), чтобы ИИ кричал "Сзади!", когда видит угрозу.
