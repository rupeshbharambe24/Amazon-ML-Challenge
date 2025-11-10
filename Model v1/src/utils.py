    import os
    import pandas as pd
    import multiprocessing
    from time import time as timer
    from tqdm import tqdm
    from functools import partial
    import requests
    from pathlib import Path

    # --- Configuration ---
    CSV_FILE_PATH = r'D:\Projects\Amazon ML Challenge\68e8d1d70b66d_student_resource\student_resource\dataset\test.csv'  # <-- CHANGE THIS TO YOUR CSV FILE PATH
    IMAGE_LINK_COLUMN = 'image_link'        # <-- ENSURE THIS MATCHES YOUR COLUMN NAME
    DOWNLOAD_FOLDER = 'test_images'   # <-- The folder where images will be saved
    NUM_WORKERS = 16                        # <-- Adjust based on your system (e.g., 4, 8, 16)# ---------------------

    def download_image(image_link: str, savefolder: str) -> str | None:
        """
        Downloads a single image link using the requests library for better robustness.

        Args:
            image_link: The URL of the image to download.
            savefolder: The local directory to save the image.

        Returns:
            The image link (str) if download failed, otherwise None.
        """
        if not isinstance(image_link, str) or not image_link.startswith('http'):
            return f"Invalid or non-HTTP link: {image_link}"

        # Use Pathlib to safely extract the filename from the URL path
        try:
            # Simple extraction of the last part of the URL path for the filename
            filename = Path(image_link).name
            if not filename:
                # Fallback if the URL ends with a slash or is strange
                filename = image_link.split('/')[-2] if len(image_link.split('/')) > 1 else 'unknown_file'
                filename = re.sub(r'[^a-zA-Z0-9\.]+', '_', filename) + '.jpg' # Simple cleaning
            
            image_save_path = os.path.join(savefolder, filename)

        except Exception as e:
            # Handle exceptions if Path() fails on a very malformed URL
            return f"Filename extraction failed for {image_link}: {e}"
            
        # Skip if the file already exists
        if os.path.exists(image_save_path):
            return None

        try:
            # Use 'requests' with a timeout for robust downloading
            response = requests.get(image_link, stream=True, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            # Write the content to the file
            with open(image_save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return None # Success

        except requests.exceptions.RequestException as ex:
            # Catches all requests-related errors (ConnectionError, Timeout, HTTPError, etc.)
            return f"Download failed for {image_link} (Requests Error): {ex}"
        except Exception as ex:
            # Catches any unexpected error (e.g., disk write error)
            return f"Download failed for {image_link} (General Error): {ex}"


    def download_images_parallel(image_links, download_folder):
        """
        Manages the parallel download process and collects failures.
        """
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
            print(f"Created download folder: {download_folder}")

        # Create a partial function to inject the savefolder argument
        download_image_partial = partial(download_image, savefolder=download_folder)
        
        # List to store all failed links/error messages
        failures = []
        
        start_time = timer()
        
        # Use multiprocessing Pool to parallelize the downloads
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            # Use pool.imap for an iterable input (image_links) and show progress with tqdm
            results_iterator = tqdm(
                pool.imap(download_image_partial, image_links),
                total=len(image_links),
                desc="Downloading Images"
            )
            
            # Collect results and check for failures
            for result in results_iterator:
                if result is not None:
                    failures.append(result)

        end_time = timer()
        print(f"\n--- Download Summary ---")
        print(f"Total links attempted: {len(image_links)}")
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        print(f"Successfully downloaded/skipped: {len(image_links) - len(failures)}")
        print(f"Total failures: {len(failures)}")

        return failures

    def download_all_images_from_csv():
        """
        Main execution function: reads the CSV and initiates the download.
        """
        try:
            # 1. Load the CSV file
            df = pd.read_csv(CSV_FILE_PATH)
            
            # 2. Extract the image links and drop any missing/empty values
            image_links = df[IMAGE_LINK_COLUMN].dropna().astype(str).tolist()
            
            if not image_links:
                print("No valid image links found in the specified column.")
                return

            print(f"Found {len(image_links)} total links to process.")
            
            # 3. Start the parallel download
            failed_downloads = download_images_parallel(image_links, DOWNLOAD_FOLDER)
            
            # 4. Report Failures
            if failed_downloads:
                print("\n--- Detailed Failures ---")
                for failure in failed_downloads:
                    print(f"FAILED: {failure}")
                
                # Optionally save failures to a file
                with open(f'{DOWNLOAD_FOLDER}/failed_downloads.txt', 'w') as f:
                    f.write('\n'.join(failed_downloads))
                print(f"\nFailure list saved to {DOWNLOAD_FOLDER}/failed_downloads.txt")
            else:
                print("\nAll images were downloaded or previously existed. Great job!")

        except FileNotFoundError:
            print(f"Error: CSV file not found at path: {CSV_FILE_PATH}")
        except KeyError:
            print(f"Error: Column '{IMAGE_LINK_COLUMN}' not found in the CSV file.")
        except Exception as e:
            print(f"An unexpected error occurred during processing: {e}")

    if __name__ == '__main__':
        # Run the main process
        download_all_images_from_csv()