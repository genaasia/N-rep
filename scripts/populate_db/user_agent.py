import os
import random
import string
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.append("../../src")
from pprint import pprint

from agent_utils import (DatabaseConnection, generate_consecutive_dates,
                         random_date_after_date, random_dates)
from text2sql.data.datasets import MysqlDataset
from utils import Config

load_dotenv()

END_DATE = datetime(2026, 1, 1)  # add to config
MAX_ORDER_COUNT = 20
APPLY_COUPON_RATE = 15

@dataclass
class UserData:
    user_id: int
    shipping_address_id: int
    joined_at: datetime


ALL_VARIANTS = []
ALL_SHIPPING_METHODS = []
ALL_COUPONS = []


class OrderManager:
    def __init__(self, db: DatabaseConnection, user_data: UserData):
        self.db = db
        self.user_data = user_data

    def _create_cart(self, created_at, updated_at):
        cart_data = [
            {
                "user_id": self.user_data.user_id,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        ]
        return self.db.insert("carts", cart_data)

    # def fill_cart(self) -> int:
    #     cart_id = self._create_or_update_cart()
    #     product_variants = self._choose_variants()
    #     self._create_cart_items(cart_id, product_variants)

    def choose_variants(self):
        weighted_numbers = [1, 1, 1, 1, 2, 2, 2, 3, 4, 5]
        product_count = random.choice(weighted_numbers)
        # product_variants = self.db.get_random_row(
        #     "product_variants", columns=["id", "product_id"], random_count=product_count
        # )
        product_variants = [random.choice(ALL_VARIANTS) for _ in range(product_count)]

        # for i, product_variant in enumerate(product_variants):
        #     product_item = self.db.get_related_rows(
        #         "products", "id", product_variant["product_id"]
        #     )
        #     assert len(product_item) == 1
        #     product_variants[i]["price"] = product_item[0]["price"]
        return product_variants

    def create_order(self, product_variants, placed_at, updated_at, cart_id) -> int:
        """Create a new order with random menu items from a random restaurant."""
        # product_variants = self._choose_variants()
        subtotal_amount = sum(
            [product_variant["price"] for product_variant in product_variants]
        )

        def generate_order_number():
            middle = str(random.randint(100, 999))
            hex_chars = "0123456789ABCDEF"
            last_part = "".join(random.choice(hex_chars) for _ in range(8))
            return f"ORD-{middle}-{last_part}"

        # shipping_method = self.db.get_random_row(
        #     "shipping_methods",
        # )[0]
        shipping_method = random.choice(ALL_SHIPPING_METHODS)

        shipping_cost = shipping_method["cost"]
        tax_amount = 0.05 * float(subtotal_amount)

        # Use coupon with APPLY_COUPON_RATE% chance
        coupon, discount_amount = None, 0
        if random.randint(1, 100) <= APPLY_COUPON_RATE:
            coupon = random.choice(ALL_COUPONS)
            # coupon = self.db.get_random_row(
            #     "coupons",
            #     foreign_key_name="discount_type",
            #     foreign_key_value="percentage",
            #     columns=["id", "value"],
            # )[0]

            discount_amount = subtotal_amount * coupon["value"] / 100

        total_amount = (
            subtotal_amount
            + Decimal(shipping_cost)
            + Decimal(tax_amount)
            - discount_amount
        )
        # placed_at = random_date_after_date(LAST_ORDER_DATE, ORDER_TIMEDELTA_RANGE)
        # updated_at = random_date_after_date(
        #     placed_at, timedelta_range=timedelta(days=5)
        # )
        status = random.choices(
            ["delivered", "cancelled", "returned"], weights=[0.97, 0.02, 0.01], k=1
        )
        order_data = [
            {
                "user_id": self.user_data.user_id,
                "order_number": generate_order_number(),
                "status": status,
                "shipping_address_id": self.user_data.shipping_address_id,
                "subtotal_amount": subtotal_amount,
                "shipping_cost": shipping_cost,
                "tax_amount": tax_amount,
                "discount_amount": discount_amount,
                "total_amount": total_amount,
                "placed_at": placed_at,
                "updated_at": updated_at,
                "cart_id": cart_id,
            }
        ]

        [order_id] = self.db.insert("orders", order_data)
        self._create_order_items(order_id, product_variants)
        if coupon:
            self.apply_coupon(
                order_id=order_id, applied_value=discount_amount, coupon_id=coupon["id"]
            )

        payment_status = random.choice(["refunded", "failed"])

        if not status == "cancelled":
            self._create_order_shipment(order_id, placed_at, shipping_method)
            payment_status = "completed"

        self._create_payment(order_id, updated_at, total_amount, status=payment_status)
        return order_id

    def apply_coupon(self, order_id, applied_value, coupon_id) -> Dict[str, Any]:
        """Apply a random coupon to the user's account."""
        order_coupon_data = [
            {
                "order_id": order_id,
                "coupon_id": coupon_id,
                "applied_value": applied_value,
            }
        ]
        self.db.insert("order_coupons", order_coupon_data)
        return order_coupon_data[0]

    def _create_order_items(
        self, order_id: int, menu_items: List[Dict[str, Any]]
    ) -> None:
        """Create order items for a given order."""
        order_items: Dict[int, Dict[str, Any]] = {}

        for item in menu_items:
            menu_item_id = item["id"]
            if menu_item_id not in order_items:
                order_items[menu_item_id] = {
                    "order_id": order_id,
                    "product_variant_id": menu_item_id,
                    "quantity": 1,
                    "unit_price": float(item["price"]),
                    "total_price": float(item["price"]),
                }
            else:
                order_items[menu_item_id]["quantity"] += 1
                order_items[menu_item_id]["price"] += float(item["price"])

        self.db.insert("order_items", list(order_items.values()))

    def _create_cart_items(
        self, cart_id: int, menu_items: List[Dict[str, Any]]
    ) -> None:
        """Create order items for a given order."""
        order_items: Dict[int, Dict[str, Any]] = {}

        for item in menu_items:
            menu_item_id = item["id"]
            if menu_item_id not in order_items:
                order_items[menu_item_id] = {
                    "cart_id": cart_id,
                    "product_variant_id": menu_item_id,
                    "quantity": 1,
                    "unit_price": float(item["price"]),
                    "total_price": float(item["price"]),
                }
            else:
                order_items[menu_item_id]["quantity"] += 1
                order_items[menu_item_id]["price"] += float(item["price"])

        self.db.insert("cart_items", list(order_items.values()))

    def _create_payment(
        self, order_id: int, order_date: datetime, amount: float, status: str
    ) -> None:
        """Create a payment record for an order."""

        def choose_payment_method():
            return random.choices(
                ["Credit Card", "Apple Pay", "PayPal", "Gift Card", "Afterpay"], weights=[0.4, 0.3, 0.2, 0.05, 0.05], k=1
            )[0]

        payment_data = [
            {
                "order_id": order_id,
                "payment_method": choose_payment_method(),
                "payment_status": status,
                "transaction_code": str(uuid.uuid4()),
                "amount": amount,
                "processed_at": order_date,
            }
        ]
        self.db.insert("payments", payment_data)

    def _create_order_shipment(
        self, order_id: int, order_date: datetime, shipping_method: Dict
    ) -> None:
        """Create a delivery record for an order."""
        letters = "".join(random.choices(string.ascii_uppercase, k=2))
        numbers = "".join(random.choices(string.digits, k=10))
        tracking_number = f"{letters}{numbers}"

        def choose_carrier():
            return random.choices(
                ["UPS", "FedEx", "USPS", "DHL"], weights=[0.4, 0.3, 0.2, 0.1], k=1
            )[0]

        def get_max_days(time_string):
            days = (
                time_string.replace("business days", "")
                .replace("business day", "")
                .strip()
            )
            if "-" in days:
                _, end = map(int, days.split("-"))
                return end
            else:
                return int(days)

        max_days = get_max_days(shipping_method["estimated_delivery_time"])
        expected_delivery_at = order_date + timedelta(days=max_days)
        delivered_at = random_date_after_date(
            expected_delivery_at - timedelta(hours=12), timedelta(hours=24)
        )
        shipped_at = (
            random_date_after_date(
                order_date, timedelta_range=timedelta(days=max_days)
            ),
        )
        shipping_method_id = shipping_method["id"]

        delivery_data = [
            {
                "order_id": order_id,
                "tracking_number": tracking_number,
                "carrier": choose_carrier(),
                "shipped_at": shipped_at,
                "expected_delivery_at": expected_delivery_at,
                "delivered_at": delivered_at,
                "shipping_method_id": shipping_method_id,
            }
        ]
        self.db.insert("order_shipments", delivery_data)


class User:
    def __init__(self, dataset: MysqlDataset, db_name: str, **user_data):
        self.data = UserData(**user_data)
        self.db = DatabaseConnection(dataset, db_name)
        self.order_manager = OrderManager(self.db, self.data)

    def _get_order(
        self, order_id: Optional[int], required_columns: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get a specific or random order for the user."""
        if not order_id:
            return self.db.get_random_row(
                "Orders", "user_id", self.data.user_id, required_columns
            )

        orders = self.db.get_related_rows(
            "Orders", "order_id", order_id, required_columns
        )
        return orders[0] if orders else None

    def make_order(self) -> int:
        """Create a new order for the user."""
        return self.order_manager.create_order()

    # def fill_cart(self) -> int:
    #     """Create a new order for the user."""
    #     return self.order_manager.fill_cart()

    def perform_user_actions(self) -> None:
        """
        Create cart with status Completed~~~
        Choose n items
        add items to cart items
        create order with same items after n seconds
        """

        # p_variants = self.order_manager.choose_variants()
        order_count = random.randint(1, MAX_ORDER_COUNT)

        card_created_at_dates = random_dates(user.data.joined_at, END_DATE, order_count)

        card_updated_at_dates = []
        order_created_at_dates = []
        payment_dates = []
        variants_list = []
        for i in range(len(card_created_at_dates)):
            card_created_at_date = card_created_at_dates[i]
            if i == len(card_created_at_dates):
                next_card_created_at_date = card_created_at_dates[i]
            else:
                next_card_created_at_date = END_DATE
            dates = generate_consecutive_dates(
                card_created_at_date, next_card_created_at_date
            )
            variants_list.append(self.order_manager.choose_variants())
            card_updated_at_dates.append(dates[1])
            order_created_at_dates.append(dates[2])
            payment_dates.append(dates[3])

        assert (
            len(card_updated_at_dates)
            == len(card_created_at_dates)
            == len(variants_list)
        )
        return (
            card_created_at_dates,
            card_updated_at_dates,
            variants_list,
            order_created_at_dates,
            payment_dates,
        )


if __name__ == "__main__":
    config = Config("config.yaml")

    ## GET DB CONNECTION
    dataset = MysqlDataset(
        os.environ.get("MYSQL_HOST"),
        os.environ.get("MYSQL_PORT"),
        os.environ.get("MYSQL_USER"),
        os.environ.get("MYSQL_PASSWORD"),
    )

    query = f"""
        SELECT
          users.id as user_id,
          addresses.id as shipping_address_id,
          users.joined_at as joined_at
        FROM {config.db_name}.addresses addresses
        JOIN {config.db_name}.users users
        ON users.id = addresses.user_id;
    """

    query_variants = """
        SELECT product_variants.*, products.price AS price
        FROM product_variants
        INNER JOIN products ON product_variants.product_id = products.id;
    """
    query_shipping_method = "SELECT * FROM shipping_methods"
    query_coupons = "SELECT * FROM coupons WHERE discount_type = 'percentage'"

    ALL_VARIANTS = dataset.query_database(config.db_name, query_variants)
    ALL_SHIPPING_METHODS = dataset.query_database(config.db_name, query_shipping_method)
    ALL_COUPONS = dataset.query_database(config.db_name, query_coupons)

    users = dataset.query_database(config.db_name, query)
    users = [User(dataset=dataset, db_name=config.db_name, **user) for user in users]

    db = DatabaseConnection(dataset, config.db_name)

    card_created_at_dates_dict = {}
    order_created_at_dates_dict = {}
    for user in tqdm(users):
        (
            card_created_at_dates,
            card_updated_at_dates,
            variants_list,
            order_created_at_dates,
            payment_dates,
        ) = user.perform_user_actions()
        for card_created_at_date, card_updated_at_date, variants in zip(
            card_created_at_dates, card_updated_at_dates, variants_list
        ):
            if card_created_at_date in card_created_at_dates_dict:
                continue
            card_created_at_dates_dict[card_created_at_date] = {}
            card_created_at_dates_dict[card_created_at_date][
                "user_id"
            ] = user.data.user_id
            card_created_at_dates_dict[card_created_at_date][
                "card_updated_at_date"
            ] = card_updated_at_date
            card_created_at_dates_dict[card_created_at_date][
                "product_variants"
            ] = variants

        for order_created_at_date, payment_date, variants, card_created_at_date in zip(
            order_created_at_dates, payment_dates, variants_list, card_created_at_dates
        ):
            if order_created_at_date in order_created_at_dates_dict:
                continue
            order_created_at_dates_dict[order_created_at_date] = {}
            order_created_at_dates_dict[order_created_at_date][
                "user_id"
            ] = user.data.user_id
            order_created_at_dates_dict[order_created_at_date]["user"] = user
            order_created_at_dates_dict[order_created_at_date][
                "payment_date"
            ] = payment_date
            order_created_at_dates_dict[order_created_at_date][
                "product_variants"
            ] = variants
            order_created_at_dates_dict[order_created_at_date][
                "card_created_at_date"
            ] = card_created_at_date

    cart_data = []
    cart_item_data = []

    for creation_date in sorted(card_created_at_dates_dict.keys()):
        cart_data.append(
            {
                "user_id": card_created_at_dates_dict[creation_date]["user_id"],
                "created_at": creation_date,
                "updated_at": card_created_at_dates_dict[creation_date][
                    "card_updated_at_date"
                ],
                "status": "completed",
            }
        )

    cart_ids = db.insert("carts", cart_data)

    for creation_date, cart_id in zip(
        sorted(card_created_at_dates_dict.keys()), cart_ids
    ):
        card_created_at_dates_dict[creation_date]["cart_id"] = cart_id
        for variant in card_created_at_dates_dict[creation_date]["product_variants"]:
            cart_item_data.append(
                {
                    "cart_id": cart_id,
                    "product_variant_id": variant["id"],
                    "quantity": 1,
                    "unit_price": float(variant["price"]),
                    "total_price": float(variant["price"]),
                }
            )

    _ = db.insert("cart_items", cart_item_data)

    for creation_date in tqdm(sorted(order_created_at_dates_dict.keys())):
        try:
            obj = order_created_at_dates_dict[creation_date]
            cart_id = card_created_at_dates_dict[obj["card_created_at_date"]]["cart_id"]
            user: User = obj["user"]
            user.order_manager.create_order(
                obj["product_variants"], creation_date, obj["payment_date"], cart_id
            )
        except Exception as e:
            print(e)
            print(f"{user.data.user_id=}")
            print()
