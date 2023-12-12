from torrentp import TorrentDownloader

def download_torrent_or_magnet(magnet_link, save_path):
    """
    Download a torrent or magnet link to a specified save path.

    Args:
        magnet_link (str): Magnet link or torrent file URL.
        save_path (str): Path where the downloaded content should be saved.

    Returns:
        None
    """
    # Create a TorrentDownloader object for handling the download
    torrent_file = TorrentDownloader(magnet_link, save_path)

    # Start the download process
    torrent_file.start_download()

    # Print a message when the download is complete
    print("Download complete!")
