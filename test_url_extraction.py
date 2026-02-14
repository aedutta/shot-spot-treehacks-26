from extract_urls import extract_urls_from_dataset_csv

urls = extract_urls_from_dataset_csv("example_dataset/brightdata_warriors_highlights.csv")
# pass urls to your embedding/clip pipeline
print(urls)