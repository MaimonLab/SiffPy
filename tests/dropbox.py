"""
Hacky ugly code to load the files from `Dropbox`, I expect to update this
at some point when I have time. For now I just want everyone to be able
to run tests on the same files!! but this doesn't even work right now...
"""
import asyncio
import aiohttp
import platform
import os
import tqdm
import re
from pathlib import Path

MAX_CONCURRENT_DOWNLOADS = 5
FOLDER_LINK = 'https://www.dropbox.com/scl/fo/nkmj9q4qh88sranihw7si/AKOy6A4aVa4vzisETgbA7mc?rlkey=qcx226wxgp6pgo611v8i7asfd&dl=0'
#DESTINATION = 'tests/data'
DESTINATION = '/Users/stephen/Desktop/Data/SICode/SiffPy/tests'

def buildurls(urls_raw, filetype='.siff'):
    """ create list of download urls - one for each file """

    # filter list for duplicates - turn it into a set and into a list again
    urls_raw = list(set(urls_raw))

    # very bad performance - n * n * n :( but very pythonic :)
    return [(
                url[:-5].split('/')[6],
                url.replace('dl=0', 'dl=1')
            ) for url in urls_raw if filetype in url]


def write_to_file(directory : str, filename : str, content):

    save_destination = Path(directory) / filename
    save_destination.mkdir(parents=True, exist_ok=True)

    with open(save_destination, 'wb') as f:
        f.write(content)

async def download(url : str, filename : str, directory : str, semaphore : asyncio.Semaphore):

    # use the semaphore to limit the concurrent connections
    async with semaphore:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                content = await resp.read()

    write_to_file(directory, filename, content)

async def get_filenames(url_overview : str):
    """ 
    Open a session, get the overview page, and extract the filenames from the html
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url_overview) as resp:
            url_overview = await resp.json()

    urls = re.findall(r'href=[\'"]?([^\'" >]+)', url_overview)
    return urls


async def asyncprogressbar(coros):
    """ visualize the progress with progressbar which shows how many files have already been downloaded """
    for f in tqdm.tqdm(asyncio.as_completed(coros), total=len(coros)):
        await f

def main():
    """ starts download of the given dropbox directory """
    if platform.system()=='Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    #asyncio.set_event_loop(asyncio.new_event_loop())
    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    # Get filenames
    urls_raw = loop.run_until_complete(get_filenames(FOLDER_LINK))
    #print(urls_raw)
    urls_download = buildurls(urls_raw)
    print(urls_download)

    # start the async download manager
    tasks = [download(url, filename, DESTINATION, sem) for filename, url in urls_download]
    #tasks = [download(FOLDER_LINK, 'downloaded', DESTINATION, sem)]
    loop.run_until_complete(asyncprogressbar(tasks))

    loop.stop()
    loop.run_forever()
    loop.close()


if __name__ == '__main__':
    main()