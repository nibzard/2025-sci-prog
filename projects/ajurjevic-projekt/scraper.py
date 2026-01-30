import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import csv

CSV_COLUMNS = [
    "listing_url","make","model","version","price","Mileage","Gearbox",
    "First registration","Fuel type","Power_kW","Power_hp","Seller",
    "Body type","Vehicle type","Drivetrain","Seats","Doors","Colour",
    "Manufacturer colour","Paint","Upholstery colour","Upholstery",
    "Comfort & Convenience","Entertainment & Media",
    "Safety & Security","Extras"
]


class AutoScout24Scraper:
    def __init__(self, make="", model="", version="", year_from="", year_to="", num_pages=1, headless=True, max_threads=4):
        self.make = make
        self.model = model
        self.version = version
        self.year_from = year_from
        self.year_to = year_to
        self.num_pages = num_pages
        self.headless = headless
        self.max_threads = max_threads
        self.listings_lock = Lock()
        self.all_listings = []

        # Build URL dynamically based on provided parameters
        url_parts = [f"https://www.autoscout24.com/lst"]
        if self.make:
            url_parts.append(f"/{self.make}")
            if self.model:
                url_parts.append(f"/{self.model}")
        
        base_url_path = "".join(url_parts)
        if self.version:
            base_url_path += f"/ve_{self.version}"
        
        self.base_url = (
            base_url_path +
            f"?atype=C&cy=I&damaged_listing=exclude&desc=0"
            f"&fregfrom={self.year_from}&fregto={self.year_to}&sort=standard"
        )

        self.listing_frame = pd.DataFrame()

        # Main browser for navigation
        self.browser = self._create_browser()

    def _create_browser(self):
        # Create and configure a Chrome webdriver instance
        options = Options()
        options.add_argument("--incognito")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-default-apps")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-logging")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-notifications")
        if self.headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-logging")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-notifications")
        options.add_argument("--log-level=3")
        options.add_argument("--silent")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        return webdriver.Chrome(options=options)

    def generate_urls(self):
        urls = [self.base_url]
        for i in range(2, self.num_pages + 1):
            urls.append(self.base_url + f"&page={i}")
        return urls

    def get_total_pages(self):
        # Extract total number of pages from pagination element
        try:
            page_indicator = self.browser.find_element(
                By.CSS_SELECTOR, 
                "li.pagination-item--page-indicator span"
            ).text.strip()
            total = int(page_indicator.split("/")[1].strip())
            return min(total, self.num_pages)  # Limit to requested pages
        except:
            return self.num_pages  # Default to requested pages

    def scrape_listing(self, listing_url, browser=None):
        # Scrape a single listing. If browser is None, creates a new one (for threading)
        if browser is None:
            browser = self._create_browser()
            created_browser = True
        else:
            created_browser = False
        
        try:
            browser.get(listing_url)
            time.sleep(2)  # wait for initial page load
            
            # Close popup in worker threads too (critical for headless mode)
            self._close_popup_with_browser(browser)
            
            # Wait for dynamic content (equipment section) to load
            time.sleep(3)
        except Exception as e:
            print(f"Invalid URL, skipping: {listing_url} -> {e}")
            if created_browser:
                browser.quit()
            return None

        data = {"listing_url": listing_url}

        # Make, Model, Version
        try:
            make_model = browser.find_element(
                By.CSS_SELECTOR,
                "div.StageTitle_makeModelContainer__RyjBP span.StageTitle_boldClassifiedInfo__sQb0l"
            ).text.strip()
            parts = make_model.split(" ", 1)
            data["make"] = parts[0] if len(parts) > 0 else None
            data["model"] = parts[1] if len(parts) > 1 else None
        except:
            data["make"] = None
            data["model"] = None

        try:
            version = browser.find_element(
                By.CSS_SELECTOR,
                "div.StageTitle_modelVersion__Yof2Z"
            ).text.strip()
            data["version"] = version
        except:
            data["version"] = None

        # Price (only numbers)
        try:
            price_elem = browser.find_element(By.CSS_SELECTOR, "span.PriceInfo_price__XU0aF")
            
            # Get text without <sup>
            price_text = browser.execute_script(
                "return arguments[0].childNodes[0].textContent;",
                price_elem
            )

            data["price"] = re.sub(r"\D", "", price_text)
        except:
            data["price"] = None

        # VehicleOverview (Mileage, Gearbox, Fuel type, etc.)
        items = browser.find_elements(By.CSS_SELECTOR, "div.VehicleOverview_itemContainer__XSLWi")
        for item in items:
            try:
                key = item.find_element(By.CSS_SELECTOR, "div.VehicleOverview_itemTitle__S2_lb").text.strip()
                value = item.find_element(By.CSS_SELECTOR, "div.VehicleOverview_itemText__AI4dA").text.strip()

                # Clean numeric values
                if key.lower() == "power":
                    # extract kW and hp separately
                    match = re.search(r"(\d+)\s*kW\s*\((\d+)\s*hp\)", value)
                    if match:
                        data["Power_kW"] = match.group(1)
                        data["Power_hp"] = match.group(2)
                    else:
                        data["Power"] = value
                elif key.lower() == "engine size":
                    cc = re.search(r"[\d,]+", value)
                    data["Engine_cc"] = cc.group(0).replace(",", "") if cc else value
                elif key.lower() == "mileage":
                    km = re.search(r"[\d,]+", value)
                    data["Mileage"] = km.group(0).replace(",", "") if km else value
                else:
                    data[key] = value
            except:
                continue

        # Basic Data (Body type, Vehicle type, Drivetrain, Seats, Doors)
        try:
            dl = browser.find_element(By.CSS_SELECTOR, "section#basic-details-section dl")
            dt_elements = dl.find_elements(By.TAG_NAME, "dt")
            dd_elements = dl.find_elements(By.TAG_NAME, "dd")
            for dt, dd in zip(dt_elements, dd_elements):
                key = dt.text.strip()
                value = dd.text.strip()
                if key in ["Body type", "Vehicle type", "Drivetrain", "Seats", "Doors"]:
                    data[key] = value
        except:
            pass

        # Environment / Fuel type
        try:
            dl = browser.find_element(By.CSS_SELECTOR, "section#environment-details-section dl")
            dt_elements = dl.find_elements(By.TAG_NAME, "dt")
            dd_elements = dl.find_elements(By.TAG_NAME, "dd")
            for dt, dd in zip(dt_elements, dd_elements):
                key = dt.text.strip()
                value = dd.text.strip()
                if key.lower() == "fuel type":  # only Fuel type
                    data["Fuel type"] = value
        except:
            pass

        # Color and Upholstery
        try:
            dl = browser.find_element(By.CSS_SELECTOR, "section#color-section dl")
            dt_elements = dl.find_elements(By.TAG_NAME, "dt")
            dd_elements = dl.find_elements(By.TAG_NAME, "dd")
            for dt, dd in zip(dt_elements, dd_elements):
                key = dt.text.strip()
                value = dd.text.strip().replace('\n', ', ')  # Replace newlines with comma-space
                data[key] = value
        except:
            pass

        # Equipment (Comfort, Entertainment, Safety, Extras)
        # Define the expected equipment sections in the correct order
        equipment_sections = [
            "Comfort & Convenience",
            "Entertainment & Media",
            "Safety & Security",
            "Extras"
        ]
        
        try:
            # Scroll to equipment section to ensure it's loaded
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            # Wait for equipment section to be present (up to 8 seconds)
            try:
                WebDriverWait(browser, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "section#equipment-section"))
                )
            except:
                pass  # Equipment section may not exist
            
            try:
                section = browser.find_element(By.CSS_SELECTOR, "section#equipment-section")
            except:
                section = None
            
            if section:
                # Click "See more" if collapsed
                try:
                    expand_btn = section.find_element(By.CSS_SELECTOR, "button.ExpandableDetailsSection_expandButton__Jir5n")
                    if expand_btn.get_attribute("aria-expanded") == "false":
                        browser.execute_script("arguments[0].scrollIntoView(true);", expand_btn)
                        time.sleep(0.5)
                        expand_btn.click()
                        time.sleep(2)  # wait for content to expand
                except:
                    pass  # no button found, already expanded

                # Now scrape dt/dd
                try:
                    dl = section.find_element(By.TAG_NAME, "dl")
                    dt_elements = dl.find_elements(By.XPATH, "./dt")

                    for dt in dt_elements:
                        key = dt.text.strip()
                        if not key:
                            continue  # skip icon-only dt
                        dd = dt.find_element(By.XPATH, "following-sibling::dd[1]")
                        items = dd.find_elements(By.TAG_NAME, "li")
                        value = ", ".join(item.text.strip().replace('\n', ' ') for item in items if item.text.strip())
                        if value:
                            data[key] = value
                except:
                    pass
            
            # Ensure all equipment sections are in data, using empty string if missing
            for section_name in equipment_sections:
                if section_name not in data:
                    data[section_name] = ""

        except Exception as e:
            # If error occurs, still add empty strings for equipment sections
            for section_name in equipment_sections:
                if section_name not in data:
                    data[section_name] = ""
        
        if created_browser:
            browser.quit()

        return data

    def _close_popup(self):
        # Close consent popup by clicking Accept All button for the main browser instance
        self._close_popup_with_browser(self.browser)

    def _close_popup_with_browser(self, browser):
        # Close consent popup by clicking Accept All button (any browser instance)
        try:
            accept_btn = WebDriverWait(browser, 3).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='as24-cmp-accept-all-button']"))
            )
            accept_btn.click()
            time.sleep(0.5)
        except:
            pass  # Popup may not appear in some cases

    def scrape(self, verbose=False):
        # Scrape listings with multithreading support
        self.all_listings = []
        
        # Go to first page
        self.browser.get(self.base_url)
        time.sleep(2)
        self._close_popup()
        
        # Get total pages available
        total_pages = self.get_total_pages()
        print(f"Will scrape {total_pages} page(s) with {self.max_threads} threads")

        for page_num in range(1, total_pages + 1):
            # Navigate to page if not first page
            if page_num > 1:
                try:
                    # Close popup before navigation
                    self._close_popup()
                    time.sleep(0.5)
                    
                    next_btn = self.browser.find_element(
                        By.XPATH,
                        "//li[contains(@class, 'prev-next')]//button[not(contains(@class, 'disabled'))]//p[contains(., 'Next')]/parent::button"
                    )
                    next_btn.click()
                    time.sleep(2)
                except Exception as e:
                    print(f"Could not navigate to page {page_num}: {e}")
                    break

            # get all listing links on the page
            listing_elements = self.browser.find_elements(By.XPATH, "//article[contains(@class, 'cldt-summary-full-item')]")
            listing_links = []
            for listing in listing_elements:
                try:
                    link_element = listing.find_element(By.TAG_NAME, "a")
                    link = link_element.get_attribute("href")
                    if link:
                        listing_links.append(link)
                except:
                    continue

            print(f"Page {page_num}: Found {len(listing_links)} listings")

            # Scrape listings with multithreading
            self._scrape_listings_multithreaded(listing_links, verbose)

        self.listing_frame = pd.DataFrame(self.all_listings)

    def _scrape_listings_multithreaded(self, listing_links, verbose=False):
        # Scrape multiple listing links concurrently using threads
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(self.scrape_listing, link): link for link in listing_links}
            
            for future in as_completed(futures):
                try:
                    details = future.result()
                    if details:
                        with self.listings_lock:
                            self.all_listings.append(details)
                        if verbose:
                            print(f"✓ Scraped: {details.get('listing_url', 'Unknown')}")
                except Exception as e:
                    link = futures[future]
                    print(f"✗ Error scraping {link}: {e}")

    def save_to_csv(self, filename="listings.csv"):
        df = self.listing_frame.copy()

        # Ensure ALL expected columns exist
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # Enforce correct order
        df = df[CSV_COLUMNS]

        df = df.fillna("").astype(str)

        file_exists = os.path.isfile(filename)

        df.to_csv(
            filename,
            mode="a",
            index=False,
            header=not file_exists,
            quoting=csv.QUOTE_ALL,
            doublequote=True,
            escapechar=None, 
            lineterminator="\n"
        )

        print(f"Data saved to {filename}")


    def quit_browser(self):
        self.browser.quit()


if __name__ == "__main__":
    make = input("Enter make (leave blank for all): ").strip() or ""
    model = input("Enter model (leave blank for all): ").strip() or ""
    version = input("Enter version (optional): ").strip() or ""
    year_from = input("Enter year from (optional): ").strip() or ""
    year_to = input("Enter year to (optional): ").strip() or ""
    num_pages_input = input("Enter number of pages (default 1): ").strip()
    num_pages = int(num_pages_input) if num_pages_input else 1
    
    headless_input = input("Run headless? (y/n, default y): ").strip().lower()
    headless = headless_input != 'n'
    
    threads_input = input("Number of threads (default 1): ").strip()
    max_threads = int(threads_input) if threads_input else 1
    
    scraper = AutoScout24Scraper(make=make, model=model, version=version, 
                                  year_from=year_from, year_to=year_to, num_pages=num_pages,
                                  headless=headless, max_threads=max_threads)
    
    try:
        scraper.scrape(verbose=True)
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user. Saving collected data...")
    except Exception as e:
        print(f"\n\nScraping stopped due to error: {e}. Saving collected data...")
    finally:
        if len(scraper.all_listings) > 0:
            scraper.save_to_csv("listings.csv")
            print(f"Saved {len(scraper.all_listings)} listings to CSV")
        else:
            print("No listings were scraped.")
        scraper.quit_browser()