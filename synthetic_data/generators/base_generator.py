"""Shared utilities for all data generators."""

import random
import string
from datetime import date, timedelta
from decimal import Decimal


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def random_decimal(lo: float, hi: float, places: int = 2) -> Decimal:
    return Decimal(str(round(random.uniform(lo, hi), places)))


def uid(prefix: str, length: int = 8) -> str:
    chars = string.ascii_uppercase + string.digits
    return f"{prefix}-{''.join(random.choices(chars, k=length))}"


def weighted_choice(choices: dict):
    """choices = {value: weight, ...}"""
    population = list(choices.keys())
    weights = list(choices.values())
    return random.choices(population, weights=weights, k=1)[0]


INDUSTRIES = [
    "SaaS", "FinTech", "HealthTech", "EdTech", "RetailTech",
    "Logistics", "Manufacturing", "Media", "Consulting", "E-commerce",
    "Real Estate", "Insurance", "Legal", "Agriculture", "Energy",
]

COUNTRIES = [
    "US", "CA", "GB", "DE", "FR", "AU", "SG", "IN", "JP", "BR",
    "NL", "SE", "NO", "CH", "AE",
]

ACCOUNT_MANAGERS = [
    "Alice Brown", "Bob Chen", "Carol Davis", "David Evans",
    "Elena Flores", "Frank Garcia", "Grace Hall", "Henry Irwin",
]
