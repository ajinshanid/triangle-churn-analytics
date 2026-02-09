import os
import logging
import random
import urllib.parse
from datetime import date
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Engine
from dotenv import load_dotenv
from faker import Faker

# Setup & Configuration 
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TriangleDataGenerator:
    """Generates synthetic customer and financial data for churn analysis."""
    
    def __init__(self, num_customers: int = 1000):
        self.fake = Faker()
        self.num_customers = num_customers
        self.customer_ids: List[str] = []
        
        Faker.seed(42)
        np.random.seed(42)
        random.seed(42)

    def generate_customers(self) -> pd.DataFrame:
        """Creates customer profiles with intentional missing age values."""
        logger.info(f"Generating {self.num_customers} customers...")
        self.customer_ids = [f"CUST_{i:06d}" for i in range(1, self.num_customers + 1)]
        
        return pd.DataFrame({
            'customer_id': self.customer_ids,
            'name': [self.fake.name() for _ in range(self.num_customers)],
            'age': [np.nan if random.random() < 0.1 else random.randint(18, 90) for _ in range(self.num_customers)],
            'province': np.random.choice(['ON', 'BC', 'AB', 'QC', 'NS'], self.num_customers),
            'join_date': [self.fake.date_between(start_date='-2y', end_date='today') for _ in range(self.num_customers)]
        })

    def generate_subscriptions(self) -> pd.DataFrame:
        """Creates card tiers and status with intentional negative fee errors."""
        logger.info("Generating subscriptions...")
        tiers = np.random.choice(['Member', 'Triangle Mastercard', 'World Elite'], self.num_customers, p=[0.6, 0.3, 0.1])
        
        def calculate_fee(tier: str) -> float:
            base_fee = 120.00 if tier == 'World Elite' else 0
            return -base_fee if random.random() < 0.05 else base_fee

        return pd.DataFrame({
            'subscription_id': [f"SUB_{i:06d}" for i in range(1, self.num_customers + 1)],
            'customer_id': self.customer_ids,
            'card_tier': tiers,
            'status': np.random.choice(['Active', 'Cancelled', 'Suspended'], self.num_customers, p=[0.7, 0.2, 0.1]),
            'annual_fee': [calculate_fee(t) for t in tiers]
        })

    def generate_payments(self, n: int = 5000) -> pd.DataFrame:
        """Generates transaction history with outliers and duplicates."""
        logger.info(f"Generating {n} payments...")
        amounts = [
            round(random.uniform(5000, 15000), 2) if random.random() < 0.01 
            else round(random.uniform(10, 500), 2) for _ in range(n)
        ]
        
        df = pd.DataFrame({
            'payment_id': [f"PAY_{i:08d}" for i in range(1, n + 1)],
            'customer_id': np.random.choice(self.customer_ids, n),
            'amount': amounts,
            'payment_date': [self.fake.date_between(start_date='-1y', end_date='today') for _ in range(n)],
            'status': np.random.choice(['Success', 'Failed', 'Declined'], n, p=[0.85, 0.10, 0.05])
        })
        
        return pd.concat([df, df.sample(n=50)], ignore_index=True)

    def generate_marketing(self, n: int = 3000) -> pd.DataFrame:
        """Generates marketing campaign engagement data."""
        logger.info(f"Generating {n} marketing touchpoints...")
        campaigns = ['Bonus CT Money', 'Winter Tire Sale', 'Back to School', 'Big Red Week']
        
        return pd.DataFrame({
            'engagement_id': [f"MKT_{i:07d}" for i in range(1, n + 1)],
            'customer_id': np.random.choice(self.customer_ids, n),
            'campaign_name': np.random.choice(campaigns, n),
            'channel': np.random.choice(['Email', 'SMS', 'App Push', 'In-Store'], n),
            'engaged': [np.nan if random.random() < 0.2 else random.choice([True, False]) for _ in range(n)],
            'engagement_date': [self.fake.date_between(start_date='-6m', end_date='today') for _ in range(n)]
        })

class DataExporter:
    """Handles data persistence to SQL databases and local files."""
    
    def __init__(self):
        self.output_dir = os.getenv("OUTPUT_DIR", "raw_data")
        os.makedirs(self.output_dir, exist_ok=True)
        self.engine = self._get_engine()

    def _get_engine(self) -> Engine:
        """Builds SQLAlchemy engine from environment variables."""
        try:
            user = os.getenv("DB_USER")
            password = urllib.parse.quote(os.getenv("DB_PASS", ""))
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            
            url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
            return create_engine(url)
        except Exception as e:
            logger.error(f"Engine creation failed: {e}")
            raise

    def save_all(self, datasets: Dict[str, pd.DataFrame]):
        """Exports all DataFrames to CSV and SQL."""
        for name, df in datasets.items():
            # Local CSV
            csv_path = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(csv_path, index=False)
            
            # Database
            try:
                df.to_sql(name, self.engine, if_exists='replace', index=False)
                logger.info(f"Successfully exported {name} to Database and CSV.")
            except Exception as e:
                logger.warning(f"Database upload failed for {name}: {e}")

# Orchestration 
def main():
    generator = TriangleDataGenerator(num_customers=1000)
    
    # Building the dataset
    data_bundle = {
        'raw_customers': generator.generate_customers(),
        'raw_subscriptions': generator.generate_subscriptions(),
        'raw_payments': generator.generate_payments(),
        'raw_marketing': generator.generate_marketing()
    }
    
    # Exporting
    exporter = DataExporter()
    exporter.save_all(data_bundle)
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()