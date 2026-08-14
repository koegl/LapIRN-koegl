from typing import Optional

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def fetch_table_static(url: str) -> Optional[pd.DataFrame]:
    response = requests.get(url)
    tables = pd.read_html(response.text)
    if tables:
        return tables[0]
    return None


def fetch_table_selenium(url: str, table_id: str) -> pd.DataFrame:
    driver = webdriver.Firefox()
    driver.get(url)
    wait = WebDriverWait(driver, 15)
    table_element = wait.until(EC.presence_of_element_located((By.ID, table_id)))
    html: str = table_element.get_attribute("outerHTML")
    driver.quit()
    df: pd.DataFrame = pd.read_html(html)[0]
    return df


def main() -> None:
    url = "https://www.codabench.org/competitions/15724/#/results-tab"
    table_id = "leaderboardTable"
    output_path = "leaderboard.csv"

    df = fetch_table_static(url)
    if df is None or df.empty:
        df = fetch_table_selenium(url, table_id)

    df.to_csv(output_path, index=False)
    print(f"Saved table with shape {df.shape} to {output_path}")


if __name__ == "__main__":
    main()
