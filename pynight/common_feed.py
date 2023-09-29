import pprint
import redis
import aioredis
import feedparser
import asyncio
from pynight.common_icecream import ic
from pynight.common_debugging import (
    debug_p,
    deus_p,
)
from pynight.async_url import (
    fetch_url,
)

##
def feed_update_check(url, key_prefix="", last_n=10):
    #: @duplicateCode/4134a31821d1a53fe62090f0649f0629
    ##
    # Connect to the Redis server (you might want to adjust the connection settings)
    r = redis.Redis(host="localhost", port=6379, db=0)

    # Fetch and parse the feed data
    feed = feedparser.parse(url)

    new_items = []

    # Use the URL as the main key for the Redis hash. You can hash it for shorter keys if needed.
    redis_hash_key = url
    redis_hash_key = f"pyfeed|{key_prefix}{redis_hash_key}"

    entries_to_check = feed.entries

    for entry in entries_to_check:
        # Use the entry's unique id or link as the key within the hash
        key = entry.id if hasattr(entry, "id") else entry.link

        # ic(deus_p, debug_p, entry['title'])

        # Check if the item already exists in the Redis hash
        if deus_p or not r.hexists(redis_hash_key, key):
            # If not, add to the new_items list
            new_items.append(entry)

            # Store the item in the Redis hash
            r.hset(redis_hash_key, key, 1)
        else:
            if debug_p:
                sys.stderr.write("entry:\n")
                pp = pprint.PrettyPrinter(stream=sys.stderr)
                pp.pprint(entry)

        if len(new_items) >= last_n:
            break

    return new_items


##
async def get_redis_pool():
    return aioredis.from_url(
        "redis://localhost",
        decode_responses=True,
    )


##
async def feed_update_async(url, redis_pool=None, key_prefix="", last_n=10):
    #: @duplicateCode/4134a31821d1a53fe62090f0649f0629
    ##
    if redis_pool is None:
        if not hasattr(feed_update_async, "_default_redis_pool"):
            feed_update_async._default_redis_pool = await get_redis_pool()
        redis_pool = feed_update_async._default_redis_pool

    r = redis_pool

    feed_data = await fetch_url(url)
    # Parse the feed data using feedparser
    feed = feedparser.parse(feed_data)

    new_items = []

    redis_hash_key = url
    redis_hash_key = f"pyfeed|{key_prefix}{redis_hash_key}"

    entries_to_check = feed.entries

    for entry in entries_to_check:
        key = entry.id if hasattr(entry, "id") else entry.link


        # Check if the item already exists in the Redis hash
        exists = await r.hexists(redis_hash_key, key)

        # ic(deus_p, debug_p, entry['title'])
        if deus_p or not exists:
            new_items.append(entry)
            await r.hset(redis_hash_key, key, 1)
        else:
            if debug_p:
                sys.stderr.write("entry:\n")
                pp = pprint.PrettyPrinter(stream=sys.stderr)
                pp.pprint(entry)

        if len(new_items) >= last_n:
            break

    return new_items


##
async def feedset_process_loop(feedset):
    if not feedset.get("enabled_p", True):
        return

    loop = asyncio.get_running_loop()
    while True:
        tasks = []
        for url in feedset["urls"]:
            # Check if update_fn is async or not
            if asyncio.iscoroutinefunction(feedset["update_fn"]):
                tasks.append(feedset["update_fn"](url))
            else:
                # If the function is synchronous, run it in the default executor (i.e., a separate thread)
                task = loop.run_in_executor(None, feedset["update_fn"], url)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Process the results
        for new_items in results:
            for item in new_items:
                feedset["processor"](item)

        # Wait for the update interval before checking again
        await asyncio.sleep(feedset["interval"])


##
