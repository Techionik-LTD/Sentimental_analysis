import csv
import time
import random
from time import sleep
from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def random_sleep():
    random_delay = random.choice([4, 5, 6, 7, 8, 9])
    return sleep(random_delay)


# Specify your CSV file paths
output_csv_file_path = r'output2.csv'

chrome_options = Options()
path = r"C:\Users\SaadSuleman\AppData\Local\Google\Chrome\User Data"

chrome_options.add_argument(f'user-data-dir={path}')
chrome_options.add_argument('profile-directory=Profile 5')
chrome_options.page_load_strategy = 'eager'
chrome_options.add_argument(
    '--user-agent=%s' % "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Initialize the Chrome driver using webdriver_manager
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
sleep(2)

# Open the CSV file in append mode
with open(output_csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Loop through pages 1 to 480
    # for i in range(1, 481):
    driver.get(
        'https://www.etsy.com/ca/listing/1688243759/baddie-bands-7?click_key=c5a40ccf06d2e7415bc61dd3128d6cd55299ecb0%3A1688243759&click_sum=12830ded&ref=sold_out-4&pro=1&frs=1')
    driver.maximize_window()
    time.sleep(5)
# Open the CSV file for writing
with open('reviewsss.csv', 'a', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)

    for i in range(1, 480):
        # Scroll down
        driver.execute_script('window.scrollBy(0,500)')

        # Find reviews
        reviews = driver.find_elements(By.CSS_SELECTOR, ".wt-text-body")
        seen_reviews = set()
        # Write reviews to CSV
        for review in reviews:
            review_text = review.text.strip()
            if review_text not in seen_reviews:
                seen_reviews.add(review_text)
                csv_writer.writerow([review_text])
                print(review_text)
                sleep(3)
        # Click on the next page link
        try:
            next_page_link = driver.find_element(By.XPATH, f"//a[@data-page='{i + 1}']")
            next_page_link.click()
            sleep(4)
        except NoSuchElementException:
            break  # Break the loop if there is no next page

# Close the Chrome driver
driver.quit()