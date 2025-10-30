import asyncio
import aiohttp
import aiofiles
import pandas as pd
import os
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from urllib.parse import quote

# НАСТРОЙКИ
INPUT_CSV = r"C:\Users\gideo\project\SUAI_RAG_BOT_SERVER\parser\out_spider\spiders\links.csv"  # путь к CSV с ссылками
DOWNLOAD_DIR = r"C:\Users\gideo\project\SUAI_RAG_BOT_SERVER\rag_sources\saved_pdf"  # папка для скачанных PDF
MAX_CONCURRENT = 12  # количество одновременных скачиваний
MAX_RETRIES = 3      # попыток при ошибке

# Создаём папку для PDF
os.makedirs(os.path.join(DOWNLOAD_DIR, "pdf"), exist_ok=True)

# Загружаем ссылки
df = pd.read_csv(INPUT_CSV, header=None, usecols=[0], dtype=str, names=["url"])
urls = df["url"].dropna().astype(str).str.strip().tolist()
urls = [u for u in urls if u.startswith(("http://", "https://"))]
urls = list(dict.fromkeys(urls))  # удаляем дубликаты

print(f"Всего ссылок для скачивания: {len(urls)}")

# Асинхронная функция скачивания PDF
async def download_pdf(session, url, idx):
    encoded_url = quote(url, safe=':/?=&')
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(encoded_url, timeout=30, ssl=False) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "")
                    if "pdf" not in content_type.lower():
                        return  # игнорируем не-PDF

                    # Генерация уникального имени файла
                    folder = "pdf"
                    safe_name = f"{folder}_{idx}_{abs(hash(url))}.pdf"
                    file_path = Path(DOWNLOAD_DIR) / folder / safe_name

                    if not file_path.exists():
                        async with aiofiles.open(file_path, 'wb') as f:
                            content = await response.read()
                            await f.write(content)
                    return  # успешно скачано
                else:
                    raise Exception(f"HTTP {response.status}")
        except:
            if attempt < MAX_RETRIES:
                await asyncio.sleep(1)
            # Если попытки закончились, просто пропускаем файл

# Главная функция
async def main():
    connector = aiohttp.TCPConnector(limit_per_host=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [download_pdf(session, url, idx+1) for idx, url in enumerate(urls)]
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Скачивание PDF"):
            await f

# Запуск скрипта
if __name__ == "__main__":
    asyncio.run(main())
