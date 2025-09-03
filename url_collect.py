from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, urlunparse

def extract_unique_sub_urls(base_url):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    
    # Initialize the Chrome driver
    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(base_url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        anchors = driver.find_elements(By.TAG_NAME, "a")
        
        unique_urls = set()
        for anchor in anchors:
            href = anchor.get_attribute("href")
            if href:
                parsed_href = urlparse(href)
                parsed_base = urlparse(base_url)
                
                if parsed_href.netloc == parsed_base.netloc and parsed_href.scheme == parsed_base.scheme:
                    path = parsed_href.path if parsed_href.path != '/' else ''
                    if path.endswith('/'):
                        path = path.rstrip('/')
                    cleaned_href = urlunparse(parsed_href._replace(fragment='', path=path))
                    unique_urls.add(cleaned_href)

        unique_urls.add(base_url)
        return unique_urls

    except Exception as e:
        print(f"Error: {e}")
        return set()
    
    finally:
        driver.quit()

if __name__ == "__main__":
    base_url = input("Enter the URL to scrape: ").strip()
    output_file = input("Enter output filename (e.g., urls3.txt): ").strip()

    if not output_file.endswith(".txt"):
        output_file += ".txt"

    unique_sub_urls = extract_unique_sub_urls(base_url)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in unique_sub_urls:
            f.write(url + "\n")
    
    print(f"✅ Unique sub-URLs found and saved to {output_file}")
