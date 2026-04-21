import asyncio
import time
from typing import Any, Dict, List

import aiohttp
import pandas as pd


# Async version of get_pageviews with better error handling
async def get_pageviews_async(
    session: aiohttp.ClientSession, url: str, start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """Get monthly pageviews for a Wikipedia article between two dates asynchronously"""

    article_title = url.split("/")[-1]
    formatted_title = article_title.replace(" ", "%20")

    api_url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        "en.wikipedia.org/all-access/all-agents/"
        f"{formatted_title}/monthly/{start_date}/{end_date}"
    )

    try:
        async with session.get(api_url) as response:
            if response.status == 200:
                data = await response.json()
                if data and "items" in data:
                    return [
                        {
                            "wikipedia_url_id": url,
                            "timestamp": item["timestamp"],
                            "views": item["views"],
                        }
                        for item in data["items"]
                    ]
                return []
            if response.status == 404:
                print(f"Article not found: {article_title}")
                return []
            print(f"HTTP {response.status} for {article_title}")
            return []
    except asyncio.TimeoutError:
        print(f"Timeout for {article_title}")
        return []
    except Exception as e:
        print(f"Error for {article_title}: {str(e)}")
        return []


# Process URLs with true concurrency using asyncio.gather
async def process_urls_concurrent(
    session: aiohttp.ClientSession,
    urls: List[str],
    max_concurrent: int = 50,
    start_date: str = "20000101",
    end_date: str = "20251231",
) -> List[Dict[str, Any]]:
    """Process URLs with controlled concurrency using asyncio.gather"""

    all_data = []

    # Process URLs in batches to control memory usage
    batch_size = max_concurrent

    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i + batch_size]

        # Create all tasks for this batch
        tasks = [
            get_pageviews_async(session, url, start_date, end_date) for url in batch_urls
        ]

        # Execute all tasks concurrently
        print(
            f"Processing batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size} ({len(batch_urls)} URLs)"
        )
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results, handling any exceptions
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"Exception for {batch_urls[j]}: {result}")
            else:
                all_data.extend(result)

        # Small delay between batches to be respectful to the API
        if i + batch_size < len(urls):
            await asyncio.sleep(0.2)

    return all_data


# Main async processing function with optimized settings
async def process_all_urls_async(
    urls: List[str], max_concurrent: int = 50
) -> List[Dict[str, Any]]:
    """Process all URLs asynchronously with optimized concurrency"""

    # Configure aiohttp session for high-performance HTTP requests
    connector = aiohttp.TCPConnector(
        limit=max_concurrent * 2,  # Total connection pool size
        limit_per_host=max_concurrent,  # Connections per host
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True,
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )

    timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=30)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"User-Agent": "Research Project (sanjay.govindan@unsw.edu.au)"},
    ) as session:
        return await process_urls_concurrent(session, urls, max_concurrent)


# Main execution
async def main():
    # Read the filtered parquet file
    print("Loading data...")
    filtered_parq = pd.read_parquet("./temporal_knowledge_wiki_merged_filtered.parquet")

    # Get unique URLs
    url_years = filtered_parq[["url"]].drop_duplicates()
    urls = url_years["url"].values.tolist()
    print(f"Processing {len(urls)} unique URLs")

    # Process all URLs asynchronously
    print("Processing with async requests (max concurrent: 50)")
    start_time = time.time()

    pageview_data = await process_all_urls_async(urls, max_concurrent=50)

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Average time per URL: {processing_time/len(urls):.2f} seconds")

    # Create DataFrame and save
    pageviews_df = pd.DataFrame(pageview_data)
    print(f"Saving {len(pageviews_df)} pageview records")
    pageviews_df.to_parquet("./wikipedia_pageviews_concurrent.parquet")

    # Print summary statistics
    print("\nSummary:")
    print(f"Total URLs processed: {len(urls)}")
    print(f"Total pageview records: {len(pageviews_df)}")
    if len(pageviews_df) > 0:
        print(
            f"Date range: {pageviews_df['timestamp'].min()} to {pageviews_df['timestamp'].max()}"
        )
        print(f"Average views per page: {pageviews_df['views'].mean():.2f}")
        print(f"Performance: {len(urls)/processing_time:.1f} URLs/second")

    return pageviews_df


# Run the async function
if __name__ == "__main__":
    result_df = asyncio.run(main())
