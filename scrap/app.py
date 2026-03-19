import pandas as pd
import asyncio
from playwright.async_api import async_playwright

INPUT_FILE = "tmdb_5000_movies.csv"
OUTPUT_FILE = "tmdb_5000_movies_with_posters.csv"

CONCURRENT_PAGES = 5


# ---------------------------
# FETCH IMAGE
# ---------------------------
async def fetch_image(page, title):
    try:
        query = f"{title} movie poster"
        url = f"https://www.google.com/search?tbm=isch&q={query}"

        await page.goto(url)
        await page.wait_for_selector("img")

        images = await page.query_selector_all("img")

        for img in images:
            src = await img.get_attribute("src")
            if src and src.startswith("http"):
                return src

    except:
        return None

    return None


# ---------------------------
# WORKER
# ---------------------------
async def worker(name, queue, results, context):
    page = await context.new_page()

    while not queue.empty():
        idx, title = await queue.get()

        print(f"[Worker {name}] {idx} → {title}")

        image_url = await fetch_image(page, title)

        results[idx] = image_url

        await page.wait_for_timeout(1000)

    await page.close()


# ---------------------------
# MAIN
# ---------------------------
async def main():
    # Load original file
    df = pd.read_csv(INPUT_FILE)

    if "title" not in df.columns:
        raise Exception("Column 'title' not found")

    titles = df["title"].fillna("").tolist()

    queue = asyncio.Queue()
    results = [None] * len(titles)

    for i, title in enumerate(titles):
        await queue.put((i, title))

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        )

        workers = [
            asyncio.create_task(worker(i, queue, results, context))
            for i in range(CONCURRENT_PAGES)
        ]

        await asyncio.gather(*workers)

        await browser.close()

    # ✅ ADD new column (does NOT remove existing ones)
    df["poster"] = results

    # ✅ Save full dataframe (all original columns + poster)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Done! All original columns preserved + 'poster' added.")


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    asyncio.run(main())