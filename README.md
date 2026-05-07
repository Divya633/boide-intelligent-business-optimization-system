# BOIDE Intelligence Engine

BOIDE is a Streamlit-based decision intelligence dashboard built on the Olist e-commerce dataset. It combines analytics, forecasting, segmentation, anomaly detection, rule-based AI insights, simulation, methodology documentation, data preview, and exportable reporting in one project.

## Project Structure

See `PROJECT_STRUCTURE.md` for a folder-by-folder map of the repository.

## Dataset

Sample datasets (`customers.csv`, `orders.csv`) are included for demonstration.

For full-scale testing, users can replace these files with larger business datasets following the same schema.
## Main Datasets Used

| Dataset | Purpose |
| `olist_orders_dataset.csv` | Order details, timestamps, and delivery status |
| `olist_order_payments_dataset.csv` | Payment values, payment methods, and installments |
| `olist_customers_dataset.csv` | Customer identifiers and location information |
| `olist_order_items_dataset.csv` | Product-level order details, prices, and freight values |
| `olist_products_dataset.csv` | Product metadata and category information |
| `olist_order_reviews_dataset.csv` | Customer review scores and feedback analysis |
| `olist_sellers_dataset.csv` | Seller details and seller locations |
| `product_category_name_translation.csv` | Translation of product categories into English |

## Main Modules

- `app.py`: Streamlit entrypoint and landing page
- `pages/`: Feature pages for each BOIDE module
- `utils/`: Shared data, UI, forecasting, simulation, and Mini-LLM logic
- `data/`: Olist CSV datasets
- `assets/`: Styles and UI assets
- `tests/`: End-to-end, unit, and integration tests
- `output/`: Generated reports and presentations
- `uml/`: Project diagrams

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the app:

```powershell
streamlit run app.py
```

## Run Tests

```powershell
python tests\test_boide.py
```

## Notes

- The app expects the Olist CSV files to remain inside `data/`.
- The Data Preview page previews uploaded CSV files but does not replace the built-in analytics dataset yet.
- Generated presentations, PDFs, and summaries are written to `output/`.
- PowerShell helpers in the project root generate review and presentation assets.
